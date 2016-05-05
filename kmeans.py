# k-means algorithm on square [0,1] x [0,1]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def closest_point(point, points):
    dist = np.sum((points - point)**2, axis = 1)
    return np.argmin(dist)


def slicedict(dict, index):
    return {k: v for k, v in dict.iteritems() if v == index}


# hard-coded data
data_size = 20
k = 4
centroids_indices = range(k)
data_indices = range(data_size)


# algorithm beginning
centroids = np.random.rand(k, 2)
old_centroids = np.zeros((k, 2))
data = np.random.rand(data_size, 2)
cluster_assignment = dict(zip(data_indices, [0] * data_size))


while True:
    for i in data_indices:
        b = cluster_assignment[i]
        cluster_assignment[i] = closest_point(data[i, :], centroids)
        c = cluster_assignment[i]
        # if(b - c != 0):
        #     print b, c

    for i in centroids_indices:
        slicing_indices = slicedict(cluster_assignment, i).keys()
        print i, slicing_indices, '\n'
        a = centroids[i]
        if not(np.array_equal(slicing_indices, [])):
            centroids[i] = np.mean(data[slicing_indices], axis = 0)
        b = centroids[i]
        if not (np.array_equal(a,b)):
            print b, c

    if np.allclose(centroids, old_centroids, rtol = 0, atol = 0):
        break

    old_centroids = centroids[:]


    print "hehe"

# ind = [0,0,1,1,2,2,2,2,2,2]
#
# plot_clusters(data,ind)
print "wyszedlem z petli"

