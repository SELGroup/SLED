import numpy as np
from scipy.spatial.distance import pdist, squareform

# construct knn affinity matrix according to self-tuning spectral clustering,
# restore edge between superpixels whoes centroids are near in space distance.
def img_knn_affinity(X, st_k, k, centroids, centroids_thresh, dist_metric="euclidean"):
    dim = X.shape[0]
    st_k = np.minimum(int(0.5*dim), st_k)
    k = np.minimum(k, st_k)
    dist_ = pdist(X, metric=dist_metric)
    pd = squareform(dist_)
    for i in range(dim):
        for j in range(i+1, dim):
            space_dist = np.sqrt(np.sum(np.square(np.array(centroids[i])-np.array(centroids[j]))))
            if space_dist > centroids_thresh:
                d = 1e20
                pd[i,j] = d
                pd[j,i] = d

    # calculate local sigma
    sigmas = np.zeros(dim)
    for i in range(dim):
        sigmas[i] = sorted(pd[i])[st_k]

    # calculate threshold for knn
    thresholds = np.zeros(dim)
    for i in range(dim):
        thresholds[i] = sorted(pd[i])[k]

    A = np.zeros([dim, dim])
    for i in range(dim):
        for j in range(i+1, dim):
            dist_ij = pd[i, j]
            w = np.exp(-1 * dist_ij ** 2 / np.clip((sigmas[i] * sigmas[j]), a_min=1e-10, a_max=None))
            if dist_ij < thresholds[i] or dist_ij < thresholds[j]:
                A[i, j] = w
                A[j, i] = w
    return A

# A naive knn graph.
def knn_affinity(X, sigma, k, centroids, centroids_thresh, dist_metric="euclidean"):
    # sigma = 30
    dim = X.shape[0]
    dist_ = pdist(X, metric=dist_metric)
    pd = squareform(dist_)
    for i in range(dim):
        for j in range(i+1, dim):
            # superpixel centroid distances calculation
            space_dist = np.sqrt(np.sum(np.square(np.array(centroids[i])-np.array(centroids[j]))))
            if space_dist > centroids_thresh:
                d = 1e20
                pd[i,j] = d
                pd[j,i] = d

    # calculate threshold for knn
    thresholds = np.zeros(dim)
    for i in range(dim):
        thresholds[i] = sorted(pd[i])[k]

    A = np.zeros([dim, dim])
    for i in range(dim):
        for j in range(i+1, dim):
            dist_ij = pd[i, j]
            w = np.exp(-1 * dist_ij ** 2 / np.clip((sigma * sigma), a_min=1e-10, a_max=None))
            if dist_ij < thresholds[i] or dist_ij < thresholds[j]:
                A[i, j] = w
                A[j, i] = w
    return A