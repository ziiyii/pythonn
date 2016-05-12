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
    k = len(centroids)
    colors = cm.Dark2(np.linspace(0, 1, k))
    centroids_indices = range(k)
    for index, col in zip(centroids_indices, colors):
        slicing_indices = slice_list(cluster_assignment, index)
        if not (np.array_equal(slicing_indices, [])):
            x = data[slicing_indices][:, 0]
            y = data[slicing_indices][:, 1]
            plt.scatter(x, y, color = col, s = [point_size]*len(x))
        plt.scatter(centroids[index, 0], centroids[index, 1], color = col,
                    s = [centroid_size], edgecolors = 'black')
    plt.draw()
    plt.pause(waiting_time)
    plt.clf()


def plot_data(data, centroids):
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, color='black', s=[point_size] * len(x))
    x = centroids[:, 0]
    y = centroids[:, 1]
    plt.scatter(x, y, color='red', s=[centroid_size] * len(x))
    plt.draw()
    plt.pause(waiting_time)
    plt.clf()


def generate_spheric_data(data_size, k):
    weights = np.random.uniform(0, 1, k)
    weights = weights / np.sum(weights)
    number_in_clusters = np.floor(np.multiply(weights, data_size))
    new_data_size = int(np.sum(number_in_clusters))

    means1 = np.random.choice(20, k, replace = False)
    means2 = np.random.choice(20, k, replace = False)
    variances = np.random.randint(1, 2, k)

    data = np.empty([new_data_size, 2])
    index_begin = 0

    for i in range(k):
        size = number_in_clusters[i]
        index_end = index_begin + size
        data[index_begin:index_end, 0] = np.random.normal(means1[i],
                                                            variances[i], size)
        data[index_begin:index_end, 1] = np.random.normal(means2[i],
                                                            variances[i], size)
        index_begin = index_end

    return data


def centroids_arrangement(data, k):
    data_size = len(data)
    data_indices = range(data_size)
    centroids = np.empty([k, 2])
    centroids[0, :] = data[np.random.choice(data_size, 1), :]

    for c in range(1, k):
        plot_data(data, centroids[:c, :])
        distr = [0] * data_size
        for i in data_indices:
            index = closest_point(data[i, :], centroids[:c, :])
            distr[i] = np.sum((data[i, :] - centroids[index, :])**2)**2
            # **2 is to make further points more probable
        distr = distr /  np.sum(distr)
        centroids[c] = data[np.random.choice(data_indices, size = 1, p = distr),
                            :]
    return centroids


def assign_points(data, centroids):
    data_size = len(data)
    data_indices = range(data_size)
    cluster_assignment = [0] * data_size
    for i in data_indices:
        cluster_assignment[i] = closest_point(data[i, :], centroids)
    return cluster_assignment


def calculate_centroids(data, centroids, cluster_assignment):
    k = len(centroids)
    centroids_indices = range(k)
    for i in centroids_indices:
        slicing_indices = slice_list(cluster_assignment, i)
        if not (np.array_equal(slicing_indices, [])):
            centroids[i] = np.mean(data[slicing_indices], axis=0)
    return centroids

def animate_algorithm(data, centroids):
    old_centroids = np.zeros((k, 2))
    while True:
        cluster_assignment = assign_points(data, centroids)

        plot_iteration(data, cluster_assignment, centroids)

        centroids = calculate_centroids(data, centroids, cluster_assignment)

        plot_iteration(data, cluster_assignment, centroids)

        if np.allclose(centroids, old_centroids, rtol = 0, atol = 0):
            break

        old_centroids = copy.copy(centroids)

    return 0


# hard-coded data
data_size = 5000
k = 7
point_size = 5
centroid_size = 60
waiting_time = 1
#np.random.seed(99)

data = generate_spheric_data(data_size, k)
# size of data changed a little bit, but order of magnitude remained the same

centroids = centroids_arrangement(data, k)

plot_data(data, centroids)

animate_algorithm(data, centroids)

SSE = 0
cluster_assignment = assign_points(data, centroids)
for i in range(k):
    slicing_indices = slice_list(cluster_assignment, i)
    if not(np.array_equal(slicing_indices, [])):
        SSE = SSE + np.sum((data[slicing_indices, :] - centroids[i, :])**2)

print SSE