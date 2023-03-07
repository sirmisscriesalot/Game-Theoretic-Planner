import numpy as np
from casadi.tools import *
from casadi import *
import matplotlib.pyplot as plt

t = MX.sym('t')

points = [[0,1], [1,1], [2,0], [1.5, -1], [0.75, 0], [0, 0.5], [-0.75, 0], [-1.5, -1], [-2,0], [0,1]]
xgrid = [i[0] for i in points]
ygrid = [i[1] for i in points]
V = np.linspace(0,1,len(points))
lutX = interpolant('LUT', 'bspline',[V],xgrid)
lutY = interpolant('LUT', 'bspline',[V],ygrid)

curve = Function('curve', [t], [blockcat([[lutX(t)], [lutY(t)]])])

# Define the arbitrary point as a variable
point = MX.sym('point', 2)

# Define the distance function as the norm of the difference
distance = norm_2(point - curve(t))

# Define the optimization problem
nlp = {'x': t, 'f': distance, 'p': point}

# Create a solver object
solver = nlpsol('solver', 'ipopt', nlp)

rand_point = [2.5, 1.5]
# Solve the problem for a given point value
point_value = np.array(rand_point)
sol = solver(p=point_value)

# Extract the optimal value of t and the nearest point on the curve
t_opt = sol['x']
point_opt = curve(t_opt)

# Print and plot the results
print(f"The nearest point on the curve to {point_value} is {point_opt} at t = {t_opt}")

#TEST PLOTTING
plotting = np.linspace(0,1,200)
plt.plot([float(lutX(i)) for i in plotting],[float(lutY(i)) for i in plotting],'b-')
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(point_opt[0],point_opt[1],'ro')
plt.plot(point_value[0],point_value[1],'go')
plt.plot(np.linspace(point_opt[0],point_value[0]).flatten(), np.linspace(point_opt[1],point_value[1]).flatten(),linestyle='dotted')
plt.xlim([-3,3])
plt.ylim([-2,2])
plt.show()