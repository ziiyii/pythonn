# k-means algorithm on square [0,1] x [0,1]

import numpy as np
import sets

def closest_point(point, points):
    dist = np.sum((points - point)**2, axis = 1)
    return np.argmin(dist)

def slicedict(dict, index):
    return {k: v for k, v in dict.iteritems() if v == index}

# hard-coded data
data_size = 10
k = 3
centroids_indices = range(k)
data_indices = range(data_size)

# algorithm beginning
centroids = np.random.rand(k, 2)
new_centroids = np.zeros((k, 2))
data = np.random.rand(data_size, 2)
cluster_assignment = dict(zip(data_indices, [0] * data_size))

while True:
    print centroids, new_centroids
    raw_input()
    for i in data_indices:
        cluster_assignment[i] = closest_point(data[i, :], centroids)

    for i in centroids_indices:
        slicing_indexes = slicedict(cluster_assignment, i).keys()
        if(np.array_equal(slicing_indexes, [])):
            new_centroids[i] = centroids[i]
        else:
            new_centroids[i] = np.mean(data[slicing_indexes], axis = 0)
        print centroids, new_centroids
        raw_input()


    if(np.allclose(centroids, new_centroids, rtol = 0, atol = 0)):
        break
    else:
        centroids = new_centroids[:]

    print "hehe"

print "wyszedlem z petli"
