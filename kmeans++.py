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


def generate_spheric_clusters(data_size, k):
    weights = np.random.uniform(0, 1, k)
    weights = weights / np.sum(weights)
    number_in_clusters = np.floor(np.multiply(weights, data_size))
    new_data_size = int(np.sum(number_in_clusters))

    means1 = np.random.choice(50, k, replace = False)
    means2 = np.random.choice(50, k, replace = False)
    variances = np.random.randint(1, 2, k)

    data = np.empty([new_data_size, 2])
    index_begin = 0

    for i in range(k):
        size = number_in_clusters[i]
        index_end = index_begin + size
        data[index_begin:index_end, 0] = np.random.normal(means1[i], variances[i], size)
        data[index_begin:index_end, 1] = np.random.normal(means2[i], variances[i], size)
        index_begin = index_end

    return data, new_data_size


# hard-coded data
data_size = 3000
k = 8
np.random.seed(6)

centroids_indices = range(k)
colors = cm.Dark2(np.linspace(0, 1, k))
waiting_time = 0.001

data, data_size = generate_spheric_clusters(data_size, k)


# algorithm beginning
np.random.seed(1)
data_indices = range(data_size)
centroids = np.empty([k, 2])
centroids[0, :] = data[np.random.choice(data_size, 1), :]

for c in range(1, k):
    distr = [0] * data_size
    for i in data_indices:
        index = closest_point(data[i, :], centroids[:c, :])
        distr[i] = np.sum((data[i, :] - centroids[index, :])**2)
    distr = distr /  np.sum(distr)
    centroids[c] = data[np.random.choice(data_indices, size = 1, p = distr), :]

old_centroids = np.zeros((k, 2))
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
