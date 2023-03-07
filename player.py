import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

###########################################################################################################################
#constant parameters
n_controls = 2
n_states = 2
N = 50
T = 0.1
d_min = 2
d_max = 100000
t_width = 2
v_max = 20
M = 10
###########################################################################################################################

#placeholder code for path, taking path as a circle
r = 20
curv = r
track_ = []
for theta in range(360):
    track_.append([r*np.cos(theta*np.pi/180), r*np.sin(theta*np.pi/180)])



class path:
    def __init__(self, track):
        self.path_ = track

    def get_S(self, pt):
        dist = []
        for path_pt in self.path_:
            dist.append((path_pt[0] - pt[0])**2 + (path_pt[1] - pt[1])**2)
        dist = np.array(dist)
        return dist.argmin()
        
    def get_T(self, theta):
        return [-1*np.sin(theta*np.pi/180), np.cos(theta*np.pi/180)]

    def get_N(self, theta):
        return [-np.cos(theta*np.pi/180), -np.sin(theta*np.pi/180)]

    def get_tau(self, theta):
        return [r*np.cos(theta*np.pi/180), r*np.sin(theta*np.pi/180)]



#################################################################################################################################
#for the lhs, minimum and maximum value of a constraint
class constraint:
   def __init__(self):
    self.lhs = []
    self.rhs_min = []
    self.rhs_max = []

class theta:
    def __init__(self, P, racetrack):
        self.racetrack = racetrack
        self.U = np.zeros([n_controls, N])
        self.P = P
        self.X = np.zeros([n_states, N + 1])

        self.X = np.zeros([n_states, N + 1])
        for k in range(N + 1):
            self.X[:, k] = P


        self.mu = np.zeros(N + 1) #lagrange multipliers
        self.S = np.zeros(N + 1)
        self.normal = np.zeros([n_states, N + 1])
        self.tangent = np.zeros([n_states, N + 1])
        self.tau = np.zeros([n_states, N + 1])

        for k in range(N + 1):
            S_ = self.racetrack.get_S(self.X[:, k])
            self.S[k] = self.racetrack.get_S(self.X[:, k])

            self.normal[:, k] = self.racetrack.get_N(S_)
            self.tangent[:, k] = self.racetrack.get_T(S_)
            self.tau[:, k] = self.racetrack.get_tau(S_)

    def update(self):
        for k in range(N):
            st = self.X[:, k] 
            cont = self.U[:, k]
            st_next = st + T*cont
            self.X[:, k + 1] = st_next

        for k in range(N + 1):
            S_ = self.racetrack.get_S(self.X[:, k])
            self.S[k] = self.racetrack.get_S(self.X[:, k])

            self.normal[:, k] = self.racetrack.get_N(S_)
            self.tangent[:, k] = self.racetrack.get_T(S_)
            self.tau[:, k] = self.racetrack.get_tau(S_)

        

class opt_prob:
    def __init__(self, theta_im, theta_il, theta_j, blocking_factor):

        self.U_ = ca.SX.sym('U_', n_controls, N) #x and y velocities
        self.U0 = theta_im.U
        self.U_min = [-v_max]*n_controls*N
        self.U_max = [v_max]*n_controls*N

        self.P = theta_im.P
        self.X_ = self.get_X_()

        self.collision_constraints = self.get_collision_constraint(theta_im, theta_j)
        self.wall_constraints = self.get_wall_constraint(theta_im)

        self.constraints = constraint()
        self.constraints.lhs = self.collision_constraints.lhs + self.wall_constraints.lhs
        self.constraints.rhs_min = self.collision_constraints.rhs_min + self.wall_constraints.rhs_max
        self.constraints.rhs_max = self.collision_constraints.rhs_min + self.wall_constraints.rhs_max

        self.objective = self.get_objective(theta_im, theta_il, theta_j)

    #state matrix for time 0 to N
    #X[k] = X[k - 1] + T*U[k - 1]
    def get_X_(self):
        X = ca.SX.sym('X', n_states, N + 1) #state matrix
        X[:, 0] = self.P[: n_states]
        for k in range(N):
            st = X[:, k] 
            cont = self.U_[:, k]
            st_next = st + T*cont
            X[:, k + 1] = st_next

        return X

    def get_collision_constraint(self, theta_im, theta_j):
        cons = constraint()
        cons_lhs = []
        for k in range(1, N + 1):
            rel_pos = theta_j.X[:, k] - theta_im.X[:, k]
            b = rel_pos/ca.norm_2(rel_pos)
            print(b)

            cons_lhs.append(ca.dot(b, theta_j.X[:, k] - self.X_[:, k]))


        cons_rhs_min = [d_min]*N
        cons_rhs_max = [d_max]*N 

        cons.lhs = cons_lhs
        cons.rhs_min = cons_rhs_min
        cons.rhs_max = cons_rhs_max

        return cons

    def get_wall_constraint(self, theta_im):
        cons = constraint()
        cons_lhs = []
        for k in range(1, N + 1):
            cons_lhs.append(ca.dot(theta_im.normal[:, k], self.X_[:, k] - theta_im.tau[:, k]))

        cons_rhs_min = [-t_width]*N 
        cons_rhs_max = [t_width]*N

        cons.lhs = cons_lhs
        cons.rhs_min = cons_rhs_min
        cons.rhs_max = cons_rhs_max

        return cons

    def get_objective(self, theta_im, theta_il, theta_j):
        sigma_num = theta_im.tangent[:, N]
        simga_den = 1 - curv*ca.dot(theta_im.X[:, N] - theta_im.tau[:, N], theta_im.normal[:, N])
        sigma = sigma_num/simga_den
        obj = ca.dot(sigma, self.X_[:, N])

        for k in range(N + 1):
            rel_pos = theta_j.X[:, k] - theta_il.X[:, k]
            b = rel_pos/ca.norm_2(rel_pos)

            if blocking_factor != 0:
                obj = obj + blocking_factor*theta_j.mu[k]*ca.dot(b, self.X_[:, k])

        return obj

def get_strategy(theta_il, theta_j, blocking_factor):
    theta_im = theta_il
    for m in range(M):
        opt_prob_i = opt_prob(theta_im, theta_il, theta_j, blocking_factor)
        prob = {'f': opt_prob_i.objective, 'x': ca.reshape(opt_prob_i.U_, n_controls*N, 1), 'g': ca.vertcat(*opt_prob_i.wall_constraints.lhs)}
        solver = ca.nlpsol('solver', 'ipopt', prob)
        sol = solver(x0 = ca.reshape(opt_prob_i.U0, n_controls*N, 1), lbx = opt_prob_i.U_min, ubx = opt_prob_i.U_max, lbg = opt_prob_i.wall_constraints.rhs_min, ubg = opt_prob_i.wall_constraints.rhs_max)


        theta_im.U = np.reshape(np.array(sol['x']),(N, n_states)).transpose()
        theta_im.update()

    return theta_im



P_i0 = [20, 0]
P_j0 = [16.77, 10.89]
track = path(track_)
blocking_factor = 0

theta_i = theta(P_i0, track)
theta_j = theta(P_j0, track)

theta_i = get_strategy(theta_i, theta_j, blocking_factor)
# print(theta_i.normal)

# for k in range(N + 1):
#     print(np.dot(theta_i.normal[:, k], theta_i.X[:, k] - theta_i.tau[:, k]))
plt.plot(theta_i.X[0], theta_i.X[1], 'r')
plt.plot(r*np.cos(np.linspace(0,360,360)*np.pi/180), r*np.sin(np.linspace(0,360,360)*np.pi/180), 'b')
plt.show()


        

