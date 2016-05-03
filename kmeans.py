import numpy as np

sample_size = 1000

def closest_point(points, point):
    dist = np.sum((points - point)**2, axis = 1)
    return np.argmin(dist)

point = np.asarray([1,2,3,4])

points = np.asarray([[1,2,3,0],[0,2,3,4],[1,2,3,4]])

print closest_point(points, point)
print points[:, 0]
