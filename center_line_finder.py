import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
from scipy.interpolate import splprep,splev
from collections import deque
 
stack = deque()
stack2 = deque()
stack3 = deque()

img = cv.imread('vegas.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
img = cv.copyMakeBorder(img, 100, 100, 100, 100, cv.BORDER_CONSTANT, None, value = 0)
ret,thresh1 = cv.threshold(img,10,255,cv.THRESH_BINARY)
edges = cv.Canny(img,100,200)
check_array = np.zeros(edges.shape)
points = edges.nonzero()
points_t = np.transpose(points)
#print(points_t)
for i in points_t:
    check_array[i[0],i[1]] = 1

randpt1 = 19851
print(randpt1)
path = []
#TEST OUT CHAINING 
stack.append(points_t[randpt1])
points_t = np.delete(points_t, randpt1, axis=0 )
count = 0
lim1 = 10464 #good enough
lim2 = 9700 #good enough 

#chaining track points on the outside 
while len(stack) > 0 and count < lim1:
    v = stack.pop()
    try:
        idxs = np.linalg.norm(points_t - v, axis=1).argmin()
    except ValueError:
        pass
    try:
        i = points_t[idxs]
    except:
        pass
    if check_array[i[0], i[1]] > 0:
        stack.append(i)
        count = count + 1
        path.append(i)
        check_array[i[0], i[1]] = 0
        points_t = np.delete(points_t, idxs, axis = 0)

#chaining track points on the inside 
randpt2 = 8222
print(randpt2)
count = 0
stack2.append(points_t[randpt2])
path2 = []
while len(stack2) > 0 and count < lim2:
    v = stack2.pop()
    try:
        idxs = np.linalg.norm(points_t - v, axis=1).argmin()
    except ValueError:
        pass
    try:
        i = points_t[idxs]
    except:
        pass
    if check_array[i[0], i[1]] > 0:
        stack2.append(i)
        count = count + 1
        path2.append(i)
        check_array[i[0], i[1]] = 0
        points_t = np.delete(points_t, idxs, axis = 0)

#Should nto be too hard to automatically separate the inner track from the outer track

path = np.array(path)
path_t = np.transpose(path)

print(path.shape)

tck, u = splprep([path_t[0],path_t[1]],s=20000) 
new_points = splev(u, tck)
new_points = np.array(new_points)
new_points_t = np.transpose(new_points)

path2 = np.array(path2)
path2_t = np.transpose(path2)

tck2, u2 = splprep([path2_t[0],path2_t[1]],s=20000) 
new_points2 = splev(u2, tck2)
new_points2 = np.array(new_points2)
new_points2_t = np.transpose(new_points2)

print("new_points",new_points.shape)

middle_pts1 = []
for i in new_points2_t:
    idxs = np.linalg.norm(new_points_t - i, axis=1).argmin()
    middle_pts1.append((new_points_t[idxs] + i)/2)

middle_pts1 = np.array(middle_pts1)

middle_pts2 = []
for i in new_points_t:
    idxs = np.linalg.norm(new_points2_t - i, axis=1).argmin()
    middle_pts2.append((new_points2_t[idxs] + i)/2)

middle_pts2 = np.array(middle_pts2)

middle_pts = []
#NEED TO DO CHAINING AGAIN I GUESS

stack3.append(middle_pts1[0])
middle_pts.append(middle_pts1[0])

middles = np.concatenate((middle_pts1,middle_pts2),axis=0)
print("middles" , middles.shape)

new_array = [tuple(row) for row in middles]
middles = np.unique(new_array,axis = 0)
middles_t = np.transpose(middles)

print("middles" , middles.shape)
iter = 17000
count = 0
while len(stack3) > 0 and count < iter:
    v = stack3.pop() 
    try:
        idxs = np.linalg.norm(middles - v, axis=1).argmin()
    except ValueError:
        pass
    try:
        i = middles[idxs]
    except:
        pass
    stack3.append(i)
    count = count + 1
    middle_pts.append(i)
    middles = np.delete(middles, idxs, axis = 0)

new_array = [tuple(row) for row in middle_pts]
middle_pts = pd.unique(new_array)
print(middle_pts)
middle_pts = [np.array(i) for i in middle_pts]
middle_pts = np.array(middle_pts)
print(middle_pts)
middle_pts_t = np.transpose(middle_pts)

# plt.gca().set_aspect('equal', adjustable='box')
# plt.scatter(middle_pts_t[0],middle_pts_t[1],s = 2, linewidths=0, color = 'purple')
# plt.plot(new_points[0], new_points[1], 'r-')
# plt.plot(new_points2[0], new_points2[1], 'y-')
# plt.show()
#the value required for s is probably related to the length of the path array
#choosing s = 5 X Lenght of path array seems to be a good estimate ?
tck, u = splprep([middle_pts_t[0],middle_pts_t[1]],s=20000) 
new_points3 = splev(u, tck)


#######################################################################################
# normal_pts1 = splev(u, tck, der=2)
# normal_pts1 = np.array(normal_pts1)
# normal_pts1_t = np.transpose(normal_pts1)

# print("normal_pts1", normal_pts1.shape)

# draw_vectors = []
# for i,j in zip(normal_pts1_t,new_points_t):
#     normz = math.sqrt(i[0]**2 + i[1]**2)
#     normvec = 50*(i/normz)
#     draw_vectors.append(j + normvec)


# draw_vectors = np.array(draw_vectors)

# print("draw vectors", draw_vectors.shape)
# for i,j in zip(draw_vectors, new_points_t):
#     plt.plot([i[0], j[0]],[i[1], j[1]], 'b-')

plt.plot(new_points[0], new_points[1], 'r-')
plt.plot(new_points2[0], new_points2[1], 'y-')
plt.plot(new_points3[0], new_points3[1], 'g-')
plt.gca().set_aspect('equal', adjustable='box')
plt.scatter(points[0], points[1], s = 2, linewidths=0, color = 'purple')
plt.scatter(path_t[0], path_t[1], s = 2, linewidths=0, color = 'blue')
plt.scatter(path2_t[0], path2_t[1], s = 2, linewidths=0, color = 'purple')
plt.scatter(middle_pts_t[0], middle_pts_t[1], s = 2, linewidths=0, color = 'cyan')
plt.show()
