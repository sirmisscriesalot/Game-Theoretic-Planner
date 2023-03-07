import math
from casadi import *
import numpy as np
from scipy.interpolate import CubicSpline

#TODO: Add typing hints
class gtp:

    def __init__(self):
        self.N = 10
        self.smp_intv = 0.1
        self.alpha = 0.1

        #following variables are used for internal iterations to solve equation 16
        self.P_int_iter = np.zeros((self.N, 2))
        self.mu_int_iter = np.zeros((self.N))
        self.beta_int_iter = np.zeros((self.N, 2))

    def int_iter_valset(self, P_set, mu_set, beta_set):
        self.P_int_iter = P_set
        self.mu_int_iter = mu_set
        self.beta_int_iter = beta_set
        
    def create_splines(self, points):
        #parameterizing with t
        self.t = np.linspace(0, 1, len(points))
        self.cs = CubicSpline(self.t, points, bc_type='periodic')

        #first derivative 
        self.dcs = self.cs.derivative(1)
        self.ddcs = self.cs.derivative(2)

    def calc_dsi_dpn(self, s):
        den = 1 - np.matmul((self.P_int_iter[self.N-1] - self.cs(s)).transpose(), self.ddcs(s))
        num = self.dcs(s).transpose()

        self.sigma = num/den

    def linearized_cost_optimization(self):
        
        U = SX.sym('u', self.N, 2)
        P = SX.sym('p', self.N, 2)

        

    


#TEST CODE 
classtest = gtp()
classtest.create_splines([[0,1], [1,1], [2,0], [1.5, -1], [0.75, 0], [0, 0.5], [-0.75, 0], [-1.5, -1], [-2,0], [0,1]])
classtest.calc_dsi_dpn(0.5)
print(classtest.sigma)