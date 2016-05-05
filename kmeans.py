# k-means algorithm on square [0,1] x [0,1]

import numpy as np
import sets

def closest_point(point, points):
    dist = np.sum((points - point)**2, axis = 1)
    return np.argmin(dist)

# hard-coded data
data_size = 100
k = 4
data_indices = range(data_size)

centroids = np.random.rand(k, 2)
data = np.random.rand(data_size, 2)
cluster_assignment = dict(zip(data_indices, [0] * data_size))

while True:
    for i in data_indices:
        cluster_assignment[i] = closest_point(data[:, i], centroids)    
