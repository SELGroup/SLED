import numpy as np
from skimage import color
from skimage import measure
import os, jpype

EPS = 1e-15

# The refinement stage of iteratively refined structural entropy.
def refinement_SE(adj, y=None):
    adj -= np.diag(np.diag(adj))
    tol = 1e-10
    max_iter = 300
    if y is None:
        n, k = adj.shape[0], 3
        y = np.random.randint(k, size=n)
    else:
        n, k = adj.shape[0], np.amax(y) + 1

    W = np.array(adj.copy(), dtype=np.float64)
    D = np.diag(np.sum(W, axis=-1, keepdims=False))
    S = np.eye(k)[y.reshape(-1)].astype(np.float64)
    volW = np.sum(W, dtype=np.float64)
    links = np.diagonal(np.matmul(np.matmul(S.T, W), S)).copy()
    degree = np.diagonal(np.clip(np.matmul(np.matmul(S.T, D), S), a_min=EPS, a_max=None)).copy()
    ses = (-links / volW) * np.log2(np.clip(degree, a_min=1e-10, a_max=None) / volW)
    z = y.copy()
    se = np.sum(ses)
    for iter_num in range(max_iter):
        for i in range(n):
            zi = z[i]
            links[zi] -=  np.matmul(W[i,:], S[:,zi]) + np.matmul(S[:,zi].T, W[:,i])
            degree[zi] -= D[i,i]
            ses[zi] = (-links[zi]/volW) * np.log2(np.clip(degree[zi], a_min=1e-10, a_max=None)/volW)
            S[i,zi] = 0
            z[i] = -1

            links_new = links.copy()
            degree_new = degree.copy()
            links_new += np.matmul(W[i,:], S) + np.matmul(W[:, i].T, S)
            degree_new += D[i,i]
            ses_new = (-links_new/volW) * np.log2(np.clip(degree_new, a_min=1e-10, a_max=None)/volW)
            delta_ses = ses_new - ses

            opt_i = np.argmax(delta_ses)

            zi = opt_i
            z[i] = zi
            S[i,zi] = 1
            links[zi] = float(links_new[zi])
            degree[zi] = float(degree_new[zi])
            ses[zi] = float(ses_new[zi])
        if np.sum(ses) - se < tol:
            break
        se = np.sum(ses)
    return z

# The merging stage of iteratively refined structural entropy.
def merging(adj, img_name, sp_scale=None):
    img_name = img_name.split('.')[0]
    if sp_scale == None:
        adj_path = f"./{img_name}_adj.txt"
        partition_path = f"./{img_name}_partition.txt"
    else:
        adj_path = f"./{img_name}_{sp_scale}_adj.txt"
        partition_path = f"./{img_name}_{sp_scale}_partition.txt"
    adj_path = os.path.abspath(adj_path)
    partition_path = os.path.abspath(partition_path)
    with open(adj_path, 'w') as f:
        f.write('{}\n'.format(int(adj.shape[0])))
        for i in range(adj.shape[0]):
            for j in range(i + 1, adj.shape[1]):
                if adj[i, j] > 0:
                    f.write('{}\t{}\t{}\n'.format(int(i + 1), int(j + 1), adj[i, j]))
    Merging = jpype.JClass("algo.Merging")
    Merging.main([adj_path, partition_path])
    if os.path.exists(adj_path):
        os.remove(adj_path)
    # read partition file
    y = np.zeros(adj.shape[0], dtype=int)
    with open(partition_path, 'r') as f:
        for comid, line in enumerate(f.readlines()):
            line = line.strip().split('\t')
            for node in line:
                y[int(node) - 1] = comid
    if os.path.exists(partition_path):
        os.remove(partition_path)
    return y