# k-means algorithm on square [0,1] x [0,1]

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def closest_point(point, points):
    dist =  np.sum((points - point)**2, axis = 1)
    return np.argmin(dist)


def slicedict(dict, index):
    return {k: v for k, v in dict.iteritems() if v == index}


def plot_iteration(data, cluster_assignment, centroids):
    for index, col in zip(centroids_indices, colors):
        slicing_indices = slicedict(cluster_assignment, index).keys()
        if not (np.array_equal(slicing_indices, [])):
            x = data[slicing_indices][:, 0]
            y = data[slicing_indices][:, 1]
            plt.scatter(x, y, color = col, s = [50]*len(x))
        plt.scatter(centroids[index, 0], centroids[index, 1], color = col, s = [200], edgecolors = 'black')


# hard-coded data
data_size = 10000
k = 16
centroids_indices = range(k)
data_indices = range(data_size)
colors = cm.Dark2(np.linspace(0, 1, k))


# algorithm beginning
centroids = np.random.rand(k, 2)
old_centroids = np.zeros((k, 2))
data = np.random.rand(data_size, 2)
cluster_assignment = dict(zip(data_indices, [0] * data_size))


while True:
    for i in data_indices:
        cluster_assignment[i] = closest_point(data[i, :], centroids)

    for i in centroids_indices:
        slicing_indices = slicedict(cluster_assignment, i).keys()
        if not(np.array_equal(slicing_indices, [])):
            centroids[i] = np.mean(data[slicing_indices], axis = 0)

    if np.allclose(centroids, old_centroids, rtol = 0, atol = 0):
        break

    old_centroids = copy.copy(centroids)

    plot_iteration(data, cluster_assignment, centroids)
    plt.draw()
    plt.pause(0.1)
    plt.clf()
