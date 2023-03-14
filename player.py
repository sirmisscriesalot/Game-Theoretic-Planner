import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

###########################################################################################################################
#constant parameters
n_controls = 2
n_states = 2
N = 50
T = 0.1
d_min = 1
d_max = 100000
t_width = 3
v_max = 10
M = 5
L = 5
control_points = [[0, 20], [20, 20], [20,0], [30, -20], [15, 0], [0, 10], [-15, 0], [-30, -20], [-40, 0], [0, 20]]
###########################################################################################################################
#placeholder code for path, taking path as a circle
r = 20
curv = r
track_ = []
for theta in range(360):
    track_.append([r*np.cos(theta*np.pi/180), r*np.sin(theta*np.pi/180)])

class newpath:
    
    def __init__(self, control_pts):
        self.t = np.array(range(len(control_pts)))
        self.cs = CubicSpline(self.t, control_points, bc_type='periodic')
        self.coeffs = self.cs.c
        self.yy = len(control_pts)
        self.t_val = 0

        self.third_powers = self.coeffs[3]
        self.second_powers = self.coeffs[2]
        self.first_powers = self.coeffs[1]
        self.no_powers = self.coeffs[0]

        self.dcs = self.cs.derivative(1)
        self.ddcs = self.cs.derivative(2)

        self.k = 4

    def find_closest_point(c, x0, x1, x2, x3, y0, y1, y2, y3):
        s = np.linspace(0,1, 100)
        point = lambda t: (x0*t**3 + x1*t**2 + x2*t + x3, y0*t**3 + y1*t**2 + y2*t + y3)
        dist = lambda x: (point(x)[0] - c[0])**2 + (point(x)[1] - c[1])**2 

        dist_list = list(map(dist,s))

        index_min = min(range(len(dist_list)), key=dist_list.__getitem__)

        return s[index_min], dist_list[index_min]
    
    def update_k(self, pt):
        self.k = self.k%(self.yy-1)
        x03 = self.third_powers[self.k][0]
        x02 = self.second_powers[self.k][0]
        x01 = self.first_powers[self.k][0]
        x00 = self.no_powers[self.k][0]

        y03 = self.third_powers[self.k][1]
        y02 = self.second_powers[self.k][1]
        y01 = self.first_powers[self.k][1]
        y00 = self.no_powers[self.k][1]

        ans0, dist0 = self.find_closest_point((pt[0], pt[1]) , x00, x01, x02, x03, y00, y01, y02, y03)

        self.nk = (self.k+1)%(self.yy-1)
        x13 = self.third_powers[self.nk][0]
        x12 = self.second_powers[self.nk][0]
        x11 = self.first_powers[self.nk][0]
        x10 = self.no_powers[self.nk][0]

        y13 = self.third_powers[self.nk][1]
        y12 = self.second_powers[self.nk][1]
        y11 = self.first_powers[self.nk][1]
        y10 = self.no_powers[self.nk][1]

        ans1, dist1 = self.find_closest_point((pt[0], pt[1]), x10, x11, x12, x13, y10, y11, y12, y13)

        nkk = (self.k+2)%(self.yy-1)
        x23 = self.third_powers[nkk][0]
        x22 = self.second_powers[nkk][0]
        x21 = self.first_powers[nkk][0]
        x20 = self.no_powers[nkk][0]

        y23 = self.third_powers[nkk][1]
        y22 = self.second_powers[nkk][1]
        y21 = self.first_powers[nkk][1]
        y20 = self.no_powers[nkk][1]

        ans2, dist2 = self.find_closest_point((pt[0], pt[1]), x20, x21, x22, x23, y20, y21, y22, y23)

        if dist2 < dist1 and dist2 < dist0:
            self.k = self.nkk
            self.t_val = self.nkk + ans2
        elif dist1 < dist0:
            self.k = self.nk
            self.t_val = self.nk + ans1
        else:
            self.t_val = self.k + ans0 

    
    def get_S(self, pt):
        self.update_k(pt)
        return self.t_val

    def get_T(self, s):
        return self.dcs(s)
    
    def get_N(self, s):
        return self.ddcs(s)


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


        self.mu = np.zeros(N) #lagrange multipliers
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
        
        self.blocking_factor = blocking_factor
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
        self.constraints.rhs_min = self.collision_constraints.rhs_min + self.wall_constraints.rhs_min
        self.constraints.rhs_max = self.collision_constraints.rhs_max + self.wall_constraints.rhs_max

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

            if self.blocking_factor != 0:
                obj = obj + self.blocking_factor*theta_j.mu[k - 1]*ca.dot(b, self.X_[:, k])

        return -obj


###################################################################################################################################################################
def get_strategy(theta_il, theta_j, blocking_factor):
    theta_im = theta_il
    for m in range(M):
        opts ={}
        opts['ipopt.print_level'] = 0
        opt_prob_i = opt_prob(theta_im, theta_il, theta_j, blocking_factor)
        prob = {'f': opt_prob_i.objective, 'x': ca.reshape(opt_prob_i.U_, n_controls*N, 1), 'g': ca.vertcat(*opt_prob_i.constraints.lhs)}
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)
        sol = solver(x0 = ca.reshape(opt_prob_i.U0, n_controls*N, 1), lbx = opt_prob_i.U_min, ubx = opt_prob_i.U_max, lbg = opt_prob_i.constraints.rhs_min, ubg = opt_prob_i.constraints.rhs_max)
        

        theta_im.U = np.reshape(np.array(sol['x']),(N, n_states)).transpose()
        theta_im.update()

    theta_im.mu = np.array(sol['lam_g'])[:N].transpose()[0]
    return theta_im

def iterated_best_response(theta_j0, P_i0, blocking_factor_i, blocking_factor_j):
    theta_il = theta(P_i0, track)
    theta_jl = theta_j0

    for l in range(L):
        theta_il = get_strategy(theta_il, theta_jl, blocking_factor_i)

        theta_jl = get_strategy(theta_jl, theta_il, blocking_factor_j)
        # print("j", theta_jl.X)

    return theta_il, theta_jl



P_i0 = [20, 0]
P_j0 = [19, -2]
track = path(track_)
blocking_factor_i = 0.2
blocking_factor_j = 0.1

for _ in range(10):
    theta_i0 = theta(P_i0, track)
    theta_j0 = theta(P_j0, track)

    theta_j0 = get_strategy(theta_j0, theta_i0, 0)

    theta_ = iterated_best_response(theta_j0, P_i0, blocking_factor_i, blocking_factor_j)
    theta_i = theta_[0]
    theta_j = theta_[1]

    #print(theta_i.X)
    plt.plot(P_i0[0], P_i0[1], 'ro')
    plt.plot(theta_i.X[0][:7], theta_i.X[1][:7], 'r')

    plt.plot(P_j0[0], P_j0[1], 'go')
    plt.plot(theta_j.X[0][:7], theta_j.X[1][:7], 'g')

    P_i0 = [theta_i.X[0][6], theta_i.X[1][6]]
    P_j0 = [theta_j.X[0][6], theta_j.X[1][6]]

plt.plot(r*np.cos(np.linspace(0,360,360)*np.pi/180), r*np.sin(np.linspace(0,360,360)*np.pi/180), 'b')
plt.show()


        
