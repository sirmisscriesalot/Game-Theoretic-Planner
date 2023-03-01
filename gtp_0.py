from casadi import *
from typing import List
from scipy.interpolate import CubicSpline
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
def arc_length_parameterization(
    points: List[List[float]], #contains coords making up centerline
):
    #parameterizing with t
    t = np.linspace(0, 1, len(points))
    cs = CubicSpline(t, points, bc_type='periodic')

    #first derivative 
    dcs = cs.derivative(1)

    #FOR TESTING
    return cs 

#FOR TESTING
OUT = arc_length_parameterization([[1,2], [4,5], [2,-1], [1,2]])
xs = np.linspace(0, 1, 100)
plt.plot(OUT(xs)[:, 0], OUT(xs)[:,1])
plt.show()






     

