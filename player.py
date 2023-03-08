import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

############################################################################################
#constant parameters
N_CONTROLS = 2
N_STATES = 2
N = 50
T = 0.1
D_MIN = 0.2
D_MAX = 100000
T_WIDTH = 3
V_MAX = 10
M = 5
L = 5
BLOCKING_FACTOR_I = 0.7
BLOCKING_FACTOR_J = 0.1

############################################################################################
#placeholder code for path, taking path as a circle
R = 20
CURV = R
track_ = []
for theta in range(360):
    track_.append([R*np.cos(theta*np.pi/180), R*np.sin(theta*np.pi/180)])

class Path:
    def __init__(self, track):
        self.path_ = track

    def get_s(self, pt):
        dist = []
        for path_pt in self.path_:
            dist.append((path_pt[0] - pt[0])**2 + (path_pt[1] - pt[1])**2)
        dist = np.array(dist)
        return dist.argmin()
        
    def get_t(self, theta):
        return [-1*np.sin(theta*np.pi/180), np.cos(theta*np.pi/180)]

    def get_n(self, theta):
        return [-np.cos(theta*np.pi/180), -np.sin(theta*np.pi/180)]

    def get_tau(self, theta):
        return [R*np.cos(theta*np.pi/180), R*np.sin(theta*np.pi/180)]


##############################################################################################
#for the lhs, minimum and maximum value of a constraint
class Constraint:
    def __init__(self):
        self.lhs = []
        self.rhs_min = []
        self.rhs_max = []

class Theta:
    def __init__(self, P, racetrack):
        self.racetrack = racetrack
        self.U = np.zeros([N_CONTROLS, N])
        self.P = P
        self.X = np.zeros([N_CONTROLS, N + 1])

        self.X = np.zeros([N_CONTROLS, N + 1])
        for k in range(N + 1):
            self.X[:, k] = P


        self.mu = np.zeros(N) #lagrange multipliers
        self.S = np.zeros(N + 1)
        self.normal = np.zeros([N_STATES, N + 1])
        self.tangent = np.zeros([N_STATES, N + 1])
        self.tau = np.zeros([N_STATES, N + 1])

        for k in range(N + 1):
            S_ = self.racetrack.get_s(self.X[:, k])
            self.S[k] = S_

            self.normal[:, k] = self.racetrack.get_n(S_)
            self.tangent[:, k] = self.racetrack.get_t(S_)
            self.tau[:, k] = self.racetrack.get_tau(S_)

    def update(self):
        for k in range(N):
            st = self.X[:, k] 
            cont = self.U[:, k]
            st_next = st + T*cont
            self.X[:, k + 1] = st_next

        for k in range(N + 1):
            S_ = self.racetrack.get_s(self.X[:, k])
            self.S[k] = self.racetrack.get_s(self.X[:, k])

            self.normal[:, k] = self.racetrack.get_n(S_)
            self.tangent[:, k] = self.racetrack.get_t(S_)
            self.tau[:, k] = self.racetrack.get_tau(S_)
       
class OptProb:
    def __init__(self, theta_im, theta_il, theta_j, blocking_factor):        
        self.blocking_factor = blocking_factor
        self.U_ = ca.SX.sym('U_', N_CONTROLS, N) #x and y velocities
        self.U0 = theta_im.U
        self.U_min = [-V_MAX]*N_CONTROLS*N
        self.U_max = [V_MAX]*N_CONTROLS*N

        self.P = theta_im.P
        self.X_ = self.get_X_()

        self.collision_constraints = self.get_collision_constraint(theta_im, theta_j)
        self.wall_constraints = self.get_wall_constraint(theta_im)

        self.constraints = Constraint()
        self.constraints.lhs = self.collision_constraints.lhs + self.wall_constraints.lhs
        self.constraints.rhs_min = (self.collision_constraints.rhs_min
                                     + self.wall_constraints.rhs_min)
        self.constraints.rhs_max = (self.collision_constraints.rhs_max
                                     + self.wall_constraints.rhs_max)

        self.objective = self.get_objective(theta_im, theta_il, theta_j)

    #state matrix for time 0 to N
    #X[k] = X[k - 1] + T*U[k - 1]
    def get_X_(self):
        X = ca.SX.sym('X', N_STATES, N + 1) #state matrix
        X[:, 0] = self.P[: N_STATES]
        for k in range(N):
            st = X[:, k] 
            cont = self.U_[:, k]
            st_next = st + T*cont
            X[:, k + 1] = st_next

        return X

    def get_collision_constraint(self, theta_im, theta_j):
        cons = Constraint()
        cons_lhs = []
        for k in range(1, N + 1):
            rel_pos = theta_j.X[:, k] - theta_im.X[:, k]
            b = rel_pos/ca.norm_2(rel_pos)

            cons_lhs.append(ca.dot(b, theta_j.X[:, k] - self.X_[:, k]))


        cons_rhs_min = [D_MIN]*N 
        cons_rhs_max = [D_MAX]*N 

        cons.lhs = cons_lhs
        cons.rhs_min = cons_rhs_min
        cons.rhs_max = cons_rhs_max

        return cons

    def get_wall_constraint(self, theta_im):
        cons = Constraint()
        cons_lhs = []
        for k in range(1, N + 1):
            cons_lhs.append(ca.dot(theta_im.normal[:, k], self.X_[:, k] - theta_im.tau[:, k]))

        cons_rhs_min = [-T_WIDTH]*N 
        cons_rhs_max = [T_WIDTH]*N

        cons.lhs = cons_lhs
        cons.rhs_min = cons_rhs_min
        cons.rhs_max = cons_rhs_max

        return cons

    def get_objective(self, theta_im, theta_il, theta_j):
        sigma_num = theta_im.tangent[:, N]
        simga_den = 1 - CURV*ca.dot(theta_im.X[:, N] - theta_im.tau[:, N], theta_im.normal[:, N])
        sigma = sigma_num/simga_den
        obj = ca.dot(sigma, self.X_[:, N])

        for k in range(N + 1):
            rel_pos = theta_j.X[:, k] - theta_il.X[:, k]
            b = rel_pos/ca.norm_2(rel_pos)

            if self.blocking_factor != 0:
                obj = obj + self.blocking_factor*theta_j.mu[k - 1]*ca.dot(b, self.X_[:, k])

        return -obj


##################################################################################
def get_strategy(theta_il, theta_j, blocking_factor):
    theta_im = theta_il
    for m in range(M):
        opt_prob_i = OptProb(theta_im, theta_il, theta_j, blocking_factor)
        prob = {'f': opt_prob_i.objective, 'x': ca.reshape(opt_prob_i.U_, N_CONTROLS*N, 1),
                'g': ca.vertcat(*opt_prob_i.constraints.lhs)}
        solver = ca.nlpsol('solver', 'ipopt', prob)
        sol = solver(x0 = ca.reshape(opt_prob_i.U0, N_CONTROLS*N, 1),
                     lbx = opt_prob_i.U_min, ubx = opt_prob_i.U_max,
                     lbg = opt_prob_i.constraints.rhs_min, ubg = opt_prob_i.constraints.rhs_max)
        
        theta_im.U = np.reshape(np.array(sol['x']),(N, N_STATES)).transpose()
        theta_im.update()

    theta_im.mu = np.array(sol['lam_g'])[:N].transpose()[0]
    return theta_im

def iterated_best_response(theta_j0, P_i0, blocking_factor_i, blocking_factor_j):
    theta_il = Theta(P_i0, track)
    theta_jl = theta_j0

    for l in range(L):
        theta_il = get_strategy(theta_il, theta_jl, blocking_factor_i)

        theta_jl = get_strategy(theta_jl, theta_il, blocking_factor_j)
        # print("j", theta_jl.X)

    return theta_il, theta_jl



P_i0 = [20, 0]
P_j0 = [19, -6]
track = Path(track_)

theta_i0 = Theta(P_i0, track)
theta_j0 = Theta(P_j0, track)

theta_j0 = get_strategy(theta_j0, theta_i0, 0)

theta_ = iterated_best_response(theta_j0, P_i0, BLOCKING_FACTOR_I, BLOCKING_FACTOR_J)
theta_i = theta_[0]
theta_j = theta_[1]

#print(theta_i.X)
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(P_i0[0], P_i0[1], 'ro')
plt.plot(theta_i.X[0], theta_i.X[1], 'r')

plt.plot(P_j0[0], P_j0[1], 'go')
plt.plot(theta_j.X[0], theta_j.X[1], 'g')

plt.plot(R*np.cos(np.linspace(0,360,360)*np.pi/180), R*np.sin(np.linspace(0,360,360)*np.pi/180), 'b')
plt.plot((R + T_WIDTH)*np.cos(np.linspace(0,360,360)*np.pi/180), (R + T_WIDTH)*np.sin(np.linspace(0,360,360)*np.pi/180), 'b', '--')
plt.plot((R - T_WIDTH)*np.cos(np.linspace(0,360,360)*np.pi/180), (R - T_WIDTH)*np.sin(np.linspace(0,360,360)*np.pi/180), 'b', '--')

plt.show()

        

