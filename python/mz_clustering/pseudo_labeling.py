from sklearn.metrics.pairwise import cosine_similarity as cosine
from sklearn.metrics.pairwise import euclidean_distances as euclidean
import numpy as np
import torch

def K_nearst_neighbor(data, dist = 'cos', K = 10):
    assert dist in ['cos', 'euclidean'], 'Dist can only be Cosine or Euclidean distance!'

    n = np.shape(data)[0]
    data = np.reshape(data, (n, -1))

    if dist == 'cos':
        sim = cosine(data, data)

    if dist == 'euclidean':
        sim = -euclidean(data, data)

    A = np.array([[0 for j in range(K)]
                for i in range(n)])

    for i in range(n):
        sim_i = sim[i]
        neighbor = np.argsort(sim_i)[-K:]
        A[i] = neighbor

    return A

def pseudo_labeling(sim_mat, ub, lb, KNN = True, A = None, index = None):

    if KNN:
        assert len(A), 'K nearest neighbor index is None!'

        pos_loc = torch.gt(sim_mat, ub).float()
        neg_loc = torch.le(sim_mat, lb).float()

    for i in range(sim_mat.shape[0]):
        inds = []
        for j in A[index[1][i],:]: #A[i,:]
            if j in index[1]:
                ind = index[1].index(j)
                inds.append(ind)
                pos_loc[i, ind] = 1
                neg_loc[i, ind] = 0
        for ind1 in inds:
            for ind2 in inds:
                pos_loc[ind1, ind2] = 1
                neg_loc[ind1, ind2] = 0


    return pos_loc, neg_loc

