import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def closest_point(point, centroids):
    dist =  np.sum((centroids - point)**2, axis = 1)
    return np.argmin(dist)


def slice_list(cluster_assignment, index):
    return [i for i, v in enumerate(cluster_assignment) if v == index]


def plot_iteration(data, cluster_assignment, centroids):
    for index, col in zip(centroids_indices, colors):
        slicing_indices = slice_list(cluster_assignment, index)
        if not (np.array_equal(slicing_indices, [])):
            x = data[slicing_indices][:, 0]
            y = data[slicing_indices][:, 1]
            plt.scatter(x, y, color = col, s = [50]*len(x))
        plt.scatter(centroids[index, 0], centroids[index, 1], color = col, s = [200], edgecolors = 'black')
    plt.draw()
    plt.pause(waiting_time)
    plt.clf()


def plot_data(data, centroids):
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, color='black', s=[50] * len(x))
    x = centroids[:, 0]
    y = centroids[:, 1]
    plt.scatter(x, y, color='green', s=[200] * len(x))
    plt.draw()
    plt.pause(waiting_time)
    plt.clf()


# hard-coded data
data_size = 100
k = 6

centroids_indices = range(k)
data_indices = range(data_size)
colors = cm.Dark2(np.linspace(0, 1, k))
waiting_time = 0.0001


# algorithm beginning
centroids = np.random.rand(k, 2)
old_centroids = np.zeros((k, 2))
data = np.random.normal(0, 1.5, size = (data_size, 2))
cluster_assignment = [0] * data_size

plot_data(data, centroids)

while True:
    for i in data_indices:
        cluster_assignment[i] = closest_point(data[i, :], centroids)

    plot_iteration(data, cluster_assignment, centroids)

    for i in centroids_indices:
        slicing_indices = slice_list(cluster_assignment, i)
        if not(np.array_equal(slicing_indices, [])):
            centroids[i] = np.mean(data[slicing_indices], axis = 0)

    plot_iteration(data, cluster_assignment, centroids)

    if np.allclose(centroids, old_centroids, rtol = 0, atol = 0):
        break

    old_centroids = copy.copy(centroids)