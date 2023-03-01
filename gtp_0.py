from casadi import *
from typing import List
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import bisect
import matplotlib.pyplot as plt
import math
import time
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
def create_splines(
    points: List[List[float]], #contains coords making up centerline
):
    #parameterizing with t
    t = np.linspace(0, 1, len(points))
    cs = CubicSpline(t, points, bc_type='periodic')

    #first derivative 
    dcs = cs.derivative(1)

    return cs, dcs 

#TODO: check the proper typing hint for splines
def calc_total_arc_length(
    dsplines, #contains the first derivative of the splines
):
    #computes the total arc length of the curve
    f = lambda x: math.sqrt(dsplines(x)[0]**2 + dsplines(x)[1]**2)
    total_arc_length, _ = quad(f, 0, 1)

    return total_arc_length

def arc_length_parameterization(
    arc_length: float, #total arc length of the spline curve
    s_i: float, #normalized to 1, arc length 
    dsplines, #contains the first derivative of the splines
):
    #function for integrating the derivative
    f = lambda x: math.sqrt(dsplines(x)[0]**2 + dsplines(x)[1]**2)
    #solves for arc length parameter
    s_i = s_i * arc_length
    a = 0
    b = 1
    res_x = bisect(lambda x: quad(f, a, x)[0] - s_i, a, b, xtol=2e-4, rtol=2e-4)

    #FOR TESTING
    return res_x

#FOR TESTING
OUT, DERIVATIVE = create_splines([[0,1], [1,1], [2,0], [1.5, -1], [0.75, 0], [0, 0.5], [-0.75, 0], [-1.5, -1], [-2,0], [-1,1], [0,1]])
length = calc_total_arc_length(DERIVATIVE)
_ = arc_length_parameterization(length, 0, DERIVATIVE)
xs = np.linspace(0, 1, 100)
print(DERIVATIVE(0.5))
print(length)
plt.plot(OUT(xs)[:, 0], OUT(xs)[:,1])
time1 = time.time()
equi_points = np.linspace(0,1,10)
x_points = []
y_points = []
s_points = []
for i in equi_points:
    s = arc_length_parameterization(length, i, DERIVATIVE)
    s_points.append(s)
    x_points.append(OUT(s)[0])
    y_points.append(OUT(s)[1])
time2 = time.time()
print(time2-time1)
print(x_points)
print(y_points)
print(s)

f = lambda x: math.sqrt(DERIVATIVE(x)[0]**2 + DERIVATIVE(x)[1]**2)
for i in range(9):
    print(quad(f, s_points[i], s_points[i+1]))

plt.plot(x_points, y_points, 'ro')
plt.xlim([-3,3])
plt.ylim([-2,2])
plt.show()






     

