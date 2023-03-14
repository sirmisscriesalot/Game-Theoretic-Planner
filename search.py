import time 
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

control_points = [[0, 1], [1, 1], [2,0], [1.5, -1], [0.75, 0], [0, 0.5], [-0.75, 0], [-1.5, -1], [-2, 0], [0, 1]]
t = np.array(range(len(control_points)))
t_rad = np.linspace(0, 2*np.pi, 15)
colors = cm.rainbow(np.linspace(0, 1, 15))
circle_points = [2.3*np.cos(t_rad), 2.3*np.sin(t_rad)]
circle_points_x = np.flip(np.roll(circle_points[0], -4))
circle_points_y = np.flip(np.roll(circle_points[1], -4))
cs = CubicSpline(t, control_points, bc_type='periodic')
coeffs = cs.c

dcs = cs.derivative(1)
ddcs = cs.derivative(2)

def find_closest_point(c, x0, x1, x2, x3, y0, y1, y2, y3):
    s = np.linspace(0,1, 100)
    point = lambda t: (x0*t**3 + x1*t**2 + x2*t + x3, y0*t**3 + y1*t**2 + y2*t + y3)
    dist = lambda x: (point(x)[0] - c[0])**2 + (point(x)[1] - c[1])**2 

    dist_list = list(map(dist,s))

    index_min = min(range(len(dist_list)), key=dist_list.__getitem__)

    return s[index_min], dist_list[index_min]

def get_tangent(s):
    return dcs(s)

def get_normal(s):
    return ddcs(s)

test_points = cs(np.linspace(0,len(control_points)))
test_points_2 = np.linspace(0 ,1)

plt.gca().set_aspect('equal', adjustable='box')
plt.plot([i[0] for i in test_points], [i[1] for i in test_points])
plt.scatter(circle_points_x, circle_points_y, c = colors)


third_powers = coeffs[3]
second_powers = coeffs[2]
first_powers = coeffs[1]
no_powers = coeffs[0]

k = 0
t_val = 0
n = len(circle_points_x)
yy = len(control_points)

time1 = time.time()

for i in range(n):
    k = k%(yy-1)
    x03 = third_powers[k][0]
    x02 = second_powers[k][0]
    x01 = first_powers[k][0]
    x00 = no_powers[k][0]

    y03 = third_powers[k][1]
    y02 = second_powers[k][1]
    y01 = first_powers[k][1]
    y00 = no_powers[k][1]

    ans0, dist0 = find_closest_point((circle_points_x[i], circle_points_y[i]) , x00, x01, x02, x03, y00, y01, y02, y03)

    nk = (k+1)%(yy-1)
    x13 = third_powers[nk][0]
    x12 = second_powers[nk][0]
    x11 = first_powers[nk][0]
    x10 = no_powers[nk][0]

    y13 = third_powers[nk][1]
    y12 = second_powers[nk][1]
    y11 = first_powers[nk][1]
    y10 = no_powers[nk][1]

    ans1, dist1 = find_closest_point((circle_points_x[i], circle_points_y[i]), x10, x11, x12, x13, y10, y11, y12, y13)

    nkk = (k+2)%(yy-1)
    x23 = third_powers[nkk][0]
    x22 = second_powers[nkk][0]
    x21 = first_powers[nkk][0]
    x20 = no_powers[nkk][0]

    y23 = third_powers[nkk][1]
    y22 = second_powers[nkk][1]
    y21 = first_powers[nkk][1]
    y20 = no_powers[nkk][1]

    ans2, dist2 = find_closest_point((circle_points_x[i], circle_points_y[i]), x20, x21, x22, x23, y20, y21, y22, y23)

    # nkkk = (k+3)%(yy-1)
    # x33 = third_powers[nkk][0]
    # x32 = second_powers[nkk][0]
    # x31 = first_powers[nkk][0]
    # x30 = no_powers[nkk][0]

    # y33 = third_powers[nkk][1]
    # y32 = second_powers[nkk][1]
    # y31 = first_powers[nkk][1]
    # y30 = no_powers[nkk][1]

    # ans3, dist3 = find_closest_point(np.array([circle_points_x[i], circle_points_y[i], x30, x31, x32, x33, y30, y31, y32, y33]))

    # if dist < dist2 and dist3 < dist1 and dist3 < dist0:
    #     k = nkkk
    #     t_val = nkkk + ans3
    #     x = cs(t_val)[0][0][0]
    #     y = cs(t_val)[0][0][1]
    #     #plt.plot(np.linspace(x,circle_points_x[i]), np.linspace(y,circle_points_y[i]), 'g--')

    if dist2 < dist1 and dist2 < dist0:
        k = nkk
        t_val = nkk + ans2
        x = cs(t_val)[0]
        y = cs(t_val)[1]
        plt.plot(np.linspace(x,circle_points_x[i]), np.linspace(y,circle_points_y[i]), 'g--')
    elif dist1 < dist0:
        k = nk
        t_val = nk + ans1
        x = cs(t_val)[0]
        y = cs(t_val)[1]
        plt.plot(np.linspace(x,circle_points_x[i]), np.linspace(y,circle_points_y[i]), 'g--')
    else:
        t_val = k + ans0 
        x = cs(t_val)[0]
        y = cs(t_val)[1]
        plt.plot(np.linspace(x,circle_points_x[i]), np.linspace(y,circle_points_y[i]), 'g--')

time2 = time.time()
print(time2-time1)

plt.show()