import numpy as np
from casadi import *

###########################################################################################################################
#placeholder code for path, taking path as a circle
r = 20
curv = r
path = []
for theta in range(360):
    path.append([r*np.cos(theta*np.pi/180) - r, r*np.sin(theta*np.pi/180)])


def closest_in_track(pt, path):
    dist = []
    for path_pt in path:
        dist.append((path_pt[0] - pt[0])**2 + (path_pt[1] - pt[1])**2)
    return dist.argmin()
    
def tangent_vec(theta):
    return [-1*np.sin(theta*np.pi/180), np.cos(theta*np.pi/180)]

def normal_vec(theta):
    return [np.cos(theta*np.pi/180), np.sin(theta*np.pi/180)]

def pos_in_path(theta):
    return [r*np.cos(theta*np.pi/180) - r, r*np.sin(theta*np.pi/180)]

#################################################################################################################################
#class for converting numerical path functions to casadi symbols
class path_operations:
    def get_tangent(self, s):
        tangent = SX.sym('tangent', 2)
        tangent_ = tangent_vec(s)
        for i in range(2):
            tangent[i] = tangent_[i]
        return tangent

    def get_normal(self, s):
        normal = SX.sym('normal', 2)
        normal_ = normal_vec(s)
        for i in range(2):
            normal[i] = normal_[i]
        return normal

    def pos_in_path_(self, s):
        pos = SX.sym('pos', 2)
        pos_ = pos_in_path(s)
        for i in range(2):
            pos[i] = pos_[i]
        return pos

#######################################################################################################################################
#for storing the numeric values of state, controls etc for a particular player
class player:
    def __init__(self, i, n_states, n_controls, N):
        self.name = i
        self.U = np.zeros([n_controls, N])
        self.P = np.zeros([n_states, 1])
        self.X = np.zeros([n_states, N + 1])
        self.lm = np.zeros(N + 1) #lagrange multipliers

#for the lhs, minimum and maximum value of a constraint
class constraint:
   def __init__(self, n_constraint):
    self.lhs = []
    self.rhs_min = []
    self.rhs_max = []

#for all operations related to creating the optimization problem for a player
class player_strategy:
    def __init__(self, player_j, player_i_prev, n_states, n_controls, N, T, d_min, w_width, blocking_factor):

        self.U = SX.sym('U', n_controls, N) #x and y velocities
        self.P = SX.sym('P', n_states, 1) #initial state
        self.X = self.get_X(n_states, N, T)
        self.s = SX.sym('s', N + 1) #arc length parameter corresponding to each state position
        self.p_ops = path_operations()
        self.collision_constraint_ = self.collision_constraint(player_j, N, d_min)
        self.wall_constraint_ = self.wall_constraint(w_width)
        self.objective_ = self.objective(player_j, player_i_prev, blocking_factor)

    #state matrix for time 0 to N
    #X[k] = X[k - 1] + T*U[k - 1]
    def get_X(self, n_states, N, T):
        X = SX.sym('X', n_states, N + 1) #state matrix
        X[:, 0] = self.P[: n_states]
        for k in range(N):
            st = X[:, k] 
            cont = self.U[:, k]
            st_next = st + T*cont
            X[:, k + 1] = st_next

        return X

    #||p_i - P_j|| >= d_min
    def collision_constraint(self, player_j, N, d_min):
        cons = constraint(N + 1)
        cons_lhs = []
        for k in range(N + 1):
            rel_pos = self.X[:, k] - player_j.X[:, k]
            cons_lhs.append(dot(rel_pos, rel_pos)/norm_2(rel_pos))


        cons_rhs_min = np.full(N+1, d_min)
        cons_rhs_max = np.full(N+1, np.inf)

        cons.lhs = cons_lhs
        cons.rhs_min = cons_rhs_min
        cons.rhs_max = cons_rhs_max

        return cons

    #|n(p_i)[p_i - tau(p_i)]| <= w_width
    def wall_constraint(self, w_width):
        cons = constraint(N + 1)
        cons_lhs = []
        for k in range(N + 1):
            normal = self.p_ops.get_normal(self.s[k])
            tau = self.p_ops.pos_in_path_(self.s[k])
            cons_lhs.append(dot(normal, self.X[:, k] - tau))

        cons_rhs_min = np.full(N+1, -w_width)
        cons_rhs_max = np.full(N+1, w_width)

        cons.lhs = cons_lhs
        cons.rhs_min = cons_rhs_min
        cons.rhs_max = cons_rhs_max

        return cons

    #function for updating s (sigma = ds/dp)
    def sigma(self, p, s_, curv):
        tangent = self.p_ops.get_tangent(s_)
        normal = self.p_ops.get_normal(s_)
        tau = self.p_ops.pos_in_path_(s_)

        den = 1 - curv*dot(p - tau, normal)
        return dot(tangent, p)/den

    def objective(self, player_j, player_i_prev, blocking_factor):   
        sigma_ = self.sigma(self.X[:, -1], self.s[-1], curv)
        obj = sigma_
        if blocking_factor != 0:
            for k in range(N + 1):
                rel_pos = player_j.X[:, k] - player_i_prev.X[:, k] 
                b = rel_pos/np.linalg.norm(rel_pos)
                obj = obj + player_j.lm[k]*dot(b, self.X[:, k])
        
        return obj

###################################################################################################################################################################

n_states = 2
n_controls = 2
N = 5
T = 0.2
v_max = 0.6
w_width = 5
d_min = 0.3
blocking_factor = 0
init_state_j = [0, 0]

player_i = player(1, n_states, n_controls, N)
player_i.P = [2, 0]
player_j = player(2, n_states, n_controls, N)
player_j.P = [0, 0]
ps_j = player_strategy(player_j, player_i, n_states, n_controls, N, T, d_min, w_width, blocking_factor)
ps_j.P = player_j.P

opt_variables = reshape(ps_j.U, 1, 2*N)
opt_prob = {'f', ps_j.objective, 'x', opt_variables}

#solver = nlpsol('solver', 'ipopt', opt_prob, opts)
#player_i_strategy = player_strategy(player_i, n_states, n_controls, N, T, d_min, w_width)
print(vertcat(*ps_j.collision_constraint_.lhs))
#print(player_i_strategy.objective(player_j, player_i, blocking_factor))


