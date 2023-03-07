import numpy as np
from casadi.tools import *
from casadi import *
import matplotlib.pyplot as plt

# Define the curve as a function of t
t = MX.sym('t')
curve = Function('curve', [t], [blockcat([[sin(t)], [cos(t)]])])

# Define the arbitrary point as a variable
point = MX.sym('point', 2)

# Define the distance function as the norm of the difference
distance = norm_2(point - curve(t))

# Define the optimization problem
nlp = {'x': t, 'f': distance, 'p': point}

# Create a solver object
solver = nlpsol('solver', 'ipopt', nlp)

# Solve the problem for a given point value
point_value = np.array([0.5, 0.5])
sol = solver(p=point_value)

# Extract the optimal value of t and the nearest point on the curve
t_opt = sol['x']
point_opt = curve(t_opt)

# Print and plot the results
print(f"The nearest point on the curve to {point_value} is {point_opt} at t = {t_opt}")
plt.plot(np.sin(np.linspace(0,2*np.pi)), np.cos(np.linspace(0,2*np.pi)))
plt.xlim([-1.2,1.2])
plt.ylim([-1.2,1.2])
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(point_opt[0],point_opt[1],'ro')
plt.plot(point_value[0],point_value[1],'go')
plt.plot(np.linspace(point_opt[0],point_value[0]).flatten(), np.linspace(point_opt[1],point_value[1]).flatten(),linestyle='dotted')
plt.show()
