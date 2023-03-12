import numpy as np
import matplotlib.pyplot as plt
from casadi import Opti
from scipy.interpolate import CubicSpline

control_points = [[0, 1], [1, 1], [2,0], [1.5, -1], [0.75, 0], [0, 0.5], [-0.75, 0], [-1.5, -1], [-2, 0], [0, 1]]
t = np.linspace(0, 1, len(control_points))
t_rad = np.linspace(0, 2*np.pi)
cs = CubicSpline(t, control_points, bc_type='periodic')
coeffs = cs.c

third_powers = coeffs[3]
second_powers = coeffs[2]
first_powers = coeffs[1]
no_powers = coeffs[0]

x_1_3 = third_powers[0][0]
x_1_2 = second_powers[0][0]
x_1_1 = first_powers[0][0]
x_1_0 = no_powers[0][0]

y_1_3 = third_powers[0][1]
y_1_2 = second_powers[0][1]
y_1_1 = first_powers[0][1]
y_1_0 = no_powers[0][1]

test_points = cs(np.linspace(0,1))
test_points_2 = np.linspace(0,0.1)

def find_closest_point(x, y, p):
    opti = Opti()
    t = opti.variable()
    X = opti.parameter(4,1)
    Y = opti.parameter(4,1)
    P = opti.parameter(2,1)

    opti.subject_to(t<=1)
    opti.subject_to(t>=0)

    opti.set_value(X,x)
    opti.set_value(Y,y)
    opti.set_value(P,p)

    opti.minimize((X(3)*t^3 + X(2)*t^2 + X(1)*t + X(0) - P(0))^2 + (Y(3)*t^3 + Y(2)*t^2 + Y(1)*t + Y(0) - P(1))^2)

    opti.solver('ipopt')
    sol = opti.solve()

    return sol.value(t)

ans = find_closest_point(np.array([x_1_3, x_1_2, x_1_1, x_1_0]), np.array([y_1_3, y_1_2, y_1_1, y_1_0]), np.array([-0.5,1]))
print(ans)




plt.gca().set_aspect('equal', adjustable='box')
plt.plot([i[0] for i in test_points], [i[1] for i in test_points])
plt.plot(3*np.cos(t_rad), 3*np.sin(t_rad))
#OK SO this is how you extract the coefficients of the spline piecewise polynomial
plt.plot([x_1_0*i**3 + x_1_1*i**2 + x_1_2*i + x_1_3 for i in test_points_2], [y_1_0*i**3 + y_1_1*i**2 + y_1_2*i + y_1_3 for i in test_points_2], 'g--')
plt.show()
