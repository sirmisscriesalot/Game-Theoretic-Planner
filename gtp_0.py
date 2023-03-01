from casadi import *
from typing import List
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import bisect
import matplotlib.pyplot as plt
import math
import numpy as np

def quadcopter_holonomic_model(
    psi: float, #yaw
    vx: float,  #x axis velocity 
    vy: float,  #y axis velocity 
    wi: float,  #angular velocity
): 
    #rotation matrix
    M = np.array(
        [[math.cos(psi), -math.sin(psi), 0],
        [math.sin(psi), math.cos(psi), 0],
        [0, 0, 1]]
    )
    #state matrix containing body frame linear velocities and angular velocities
    S = np.array(
        [[vx],
        [vy],
        [wi]]
    )

    P = np.matmul(M,S)
    return P

#TODO: clean up the typing hint for list of coordinates 
#TODO: following code is very messy to be actually used clean it up 
def arc_length_parameterization(
    points: List[List[float]], #contains coords making up centerline
    s_i: float, #normalized to 1, arc length 
):
    #parameterizing with t
    t = np.linspace(0, 1, len(points))
    cs = CubicSpline(t, points, bc_type='periodic')

    #first derivative 
    dcs = cs.derivative(1)

    #computes the total arc length of the curve
    f = lambda x: math.sqrt(dcs(x)[0]**2 + dcs(x)[1]**2)
    total_arc_length, _ = quad(f, 0, 1)

    #solves for arc length parameter
    s_i = s_i * total_arc_length
    a = 0
    b = 1
    res_x = bisect(lambda x: quad(f, a, x)[0] - s_i, a, b)

    #FOR TESTING
    return cs, dcs, total_arc_length, res_x

#FOR TESTING
OUT,DERIVATIVE,length,_ = arc_length_parameterization([[0,1], [1,1], [2,0], [1.5, -1], [0.75, 0], [0, 0.5], [-0.75, 0], [-1.5, -1], [-2,0], [-1,1], [0,1]], 0)
xs = np.linspace(0, 1, 100)
print(DERIVATIVE(0.5))
print(length)
plt.plot(OUT(xs)[:, 0], OUT(xs)[:,1])

equi_points = np.linspace(0,1,10)
x_points = []
y_points = []
for i in equi_points:
    OUTPUT,_,_,s = arc_length_parameterization([[0,1], [1,1], [2,0], [1.5, -1], [0.75, 0], [0, 0.5], [-0.75, 0], [-1.5, -1], [-2,0], [-1,1], [0,1]], i)
    x_points.append(OUT(s)[0])
    y_points.append(OUT(s)[1])

plt.plot(x_points, y_points, 'ro')
plt.xlim([-3,3])
plt.ylim([-2,2])
plt.show()






     

