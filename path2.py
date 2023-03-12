import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from scipy.interpolate import CubicSpline

control_points = [[0, 1], [1, 1], [2,0], [1.5, -1], [0.75, 0], [0, 0.5], [-0.75, 0], [-1.5, -1], [-2, 0], [0, 1]]
t = np.array(range(len(control_points)))
t_rad = np.linspace(0, 2*np.pi)
cs = CubicSpline(t, control_points, bc_type='periodic')
coeffs = cs.c

third_powers = coeffs[3]
second_powers = coeffs[2]
first_powers = coeffs[1]
no_powers = coeffs[0]

k = 8
x_1_3 = third_powers[k][0]
x_1_2 = second_powers[k][0]
x_1_1 = first_powers[k][0]
x_1_0 = no_powers[k][0]

print(x_1_0)

y_1_3 = third_powers[k][1]
y_1_2 = second_powers[k][1]
y_1_1 = first_powers[k][1]
y_1_0 = no_powers[k][1]

test_points = cs(np.linspace(0,len(control_points)))
test_points_2 = np.linspace(0 ,1)

def find_closest_point(params):
    t = ca.MX.sym('t')
    P = ca.MX.sym('P', 10) #all the parameters 
    
    Z = ca.blockcat([[P[2]*t**3 + P[3]*t**2 + P[4]*t + P[5]], [P[6]*t**3 + P[7]*t**2 + P[8]*t + P[9]]])
    curve = ca.Function('curve', [t], [Z])
    distance = ca.norm_2(P[:2] - curve(t))
    nlp = {'x': t, 'f': distance, 'p': P}

    solver = ca.nlpsol('solver', 'ipopt', nlp)
    sol = solver(x0=0.5, lbx = 0, ubx = 1, p=params)
    t_opt = sol['x']

    return t_opt

ans = find_closest_point(np.array([-1.5, 1.5, x_1_0, x_1_1, x_1_2, x_1_3, y_1_0, y_1_1, y_1_2, y_1_3]))
print(ans)
# x = cs(k+ans)[0][0][0]
# y = cs(k+ans)[0][0][1]

ans = float(ans)
print(ans)
x = x_1_0*ans**3 + x_1_1*ans**2 + x_1_2*ans + x_1_3
y = y_1_0*ans**3 + y_1_1*ans**2 + y_1_2*ans + y_1_3

plt.plot(x, y, 'ro')
plt.plot([-1.5],[1.5], 'ro')
plt.plot(np.linspace(x,-1.5), np.linspace(y,1.5), 'g--')
plt.gca().set_aspect('equal', adjustable='box')
plt.plot([i[0] for i in test_points], [i[1] for i in test_points])
#OK SO this is how you extract the coefficients of the spline piecewise polynomial
plt.plot([x_1_0*i**3 + x_1_1*i**2 + x_1_2*i + x_1_3 for i in test_points_2], [y_1_0*i**3 + y_1_1*i**2 + y_1_2*i + y_1_3 for i in test_points_2], 'g--')
plt.show()
