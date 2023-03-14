import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# t = ca.MX.sym('t')

points = [[0,1], [1,1], [2,0], [1.5, -1], [0.75, 0], [0, 0.5], [-0.75, 0], [-1.5, -1], [-2,0], [0,1]]
# xgrid = [i[0] for i in points]
# ygrid = [i[1] for i in points]
# V = np.linspace(0,1,len(points))
# lutX = ca.interpolant('LUT', 'bspline',[V],xgrid)
# lutY = ca.interpolant('LUT', 'bspline',[V],ygrid)

# curve = ca.Function('curve', [t], [ca.blockcat([[lutX(t)], [lutY(t)]])])

# # Define the arbitrary point as a variable
# point = ca.MX.sym('point', 2)

# # Define the distance function as the norm of the difference
# distance = ca.norm_2(point - curve(t))

# # Define the optimization problem
# nlp = {'x': t, 'f': distance, 'p': point}

# # Create a solver object
# solver = ca.nlpsol('solver', 'ipopt', nlp)

rand_point = [2.5, 1]
# # Solve the problem for a given point value
point_value = np.array(rand_point)
# sol = solver(p=point_value)

# # Extract the optimal value of t and the nearest point on the curve
# t_opt = sol['x']
# point_opt = curve(t_opt)

# # Print and plot the results
# print(f"The nearest point on the curve to {point_value} is {point_opt} at t = {t_opt}")

#writing functions 

###########################################################################################
#CUSTOM FUNCTION TEST 

class MyCallback(ca.Callback):
  def __init__(self, name, d, opts={}):
    ca.Callback.__init__(self)
    self.d = d
    self.construct(name, opts)

  # Number of inputs and outputs
  def get_n_in(self): return 1
  def get_n_out(self): return 1

  # Initialize the object
  def init(self):
     print('initializing object')

  # Evaluate numerically
  def eval(self, arg):
    x = arg[0]
    f = np.sin(self.d*x)
    return [f]
  
###########################################################################################

class curve_params: 
    def __init__(self, curve_points):

        #FOR SOLVING OPTIMIZATION PROBLEM WIITH CASADI 
        t = ca.MX.sym('t')
        xgrid = [i[0] for i in curve_points]
        ygrid = [i[1] for i in curve_points]

        V = np.linspace(0,1,len(curve_points))
        self.lutX = ca.interpolant('LUT', 'bspline',[V],xgrid)
        self.lutY = ca.interpolant('LUT', 'bspline',[V],ygrid)

        

        Z = ca.blockcat([[self.lutX(t)], [self.lutY(t)]])

        self.curve = ca.Function('curve', [t], [Z])
        point = ca.MX.sym('point', 2)
        distance = ca.norm_2(point - self.curve(t))
        nlp = {'x': t, 'f': distance, 'p': point}

        self.solver = ca.nlpsol('solver', 'ipopt', nlp)

        #NEEDED TO FIND TANGENT AND NORMAL
        f_T = ca.jacobian(Z,t)
        f_N = ca.jacobian(f_T, t)

        self.curve_T = ca.Function('curve_tangent', [t], [f_T])
        self.curve_N = ca.Function('curve_normal', [t], [f_N])

    def find_t(self, point):
        point_value = np.array(point)
        sol = self.solver(p=point_value)
        self.t_opt = sol['x']
        self.point_opt = self.curve(self.t_opt)

    def get_T(self):
        return self.curve_T(self.t_opt)

    def get_N(self):
        return self.curve_N(self.t_opt)

    def get_k(self):
        pass



racetrack = curve_params(points) 
racetrack.find_t(point_value)
T = racetrack.get_T()
N = racetrack.get_N()
T_normalized = T/np.linalg.norm(T)
N_normalized = N/np.linalg.norm(N)

print(ca.dot(T_normalized,N_normalized))

#TEST PLOTTING
plotting = np.linspace(0,1,200)
plt.plot([float(racetrack.lutX(i)) for i in plotting],[float(racetrack.lutY(i)) for i in plotting],'b-')
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(racetrack.point_opt[0],racetrack.point_opt[1],'ro')
plt.plot(point_value[0],point_value[1],'go')
plt.plot(np.linspace(racetrack.point_opt[0],point_value[0]).flatten(), np.linspace(racetrack.point_opt[1],point_value[1]).flatten(),linestyle='dotted')
plt.plot(np.linspace(racetrack.point_opt[0],racetrack.point_opt[0]+T_normalized[0]).flatten(), np.linspace(racetrack.point_opt[1],racetrack.point_opt[1]+T_normalized[1]).flatten(),linestyle='dotted')
plt.plot(np.linspace(racetrack.point_opt[0],racetrack.point_opt[0]+N_normalized[0]).flatten(), np.linspace(racetrack.point_opt[1],racetrack.point_opt[1]+N_normalized[1]).flatten(),linestyle='dotted')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.show()