import numpy as np
from scipy.linalg import svd
from copy import deepcopy
from collections import Counter
from pdb import set_trace


def pca(X, n_components=5):
    #  Use svd to perform PCA on X
    #  Inputs:
    #     X: input matrix
    #     n_components: number of principal components to keep
    #  Output:
    #     principal_components: the top n_components principal_components
    #     X_pca = X.dot(principal_components)

    U, s, Vh = svd(X)
    v = np.transpose(Vh)
    # print(v)

    # Write your own code
    principal_components = v[:, :n_components]
    return principal_components


def vector_norm(x, norm="Min-Max"):
    # Calculate the normalized vector
    # Input x: 1-d np.array
    if norm == "Min-Max":
        x_norm = [(i - min(x)) / (max(x) - min(x)) for i in x]
    elif norm == "L1":
        x_sum = [sum(abs(i)) for i in x]
        x_norm = [i / x_sum for i in x]
    elif norm == "L2":
        x_norm = [i / (sum(x ** 2) ** 0.5) for i in x]
    elif norm == "Standard_Score":
        x_norm = [(i - np.mean(x)) / np.std(x) for i in x]
    else:
        raise Exception("Unknown normlization.")
    return x_norm


def normalize(X, norm="Min-Max", axis=1):
    #  Inputs:
    #     X: input matrix
    #     norm = {"L1", "L2", "Min-Max", "Standard_Score"}
    #     axis = 0: normalize rows
    #     axis = 1: normalize columns
    #  Output:
    #     X_norm: normalized matrix (numpy.array)

    X_norm = deepcopy(np.asarray(X))
    m, n = X_norm.shape
    if axis == 1:
        for col in range(n):
            X_norm[:, col] = vector_norm(X_norm[:, col], norm=norm)
    elif axis == 0:
        X_norm = np.array([vector_norm(X_norm[i], norm=norm) for i in range(m)])
    else:
        raise Exception("Unknown axis.")
    return X_norm


def stratified_sampling(y, ratio, replace=True):
    #  Inputs:
    #     y: a 1-d array of class labels
    #     0 < ratio < 1: number of samples = len(y) * ratio
    #     replace = True: sample with replacement
    #     replace = False: sample without replacement
    #  Output:
    #     sample: indices of stratified sampled points
    #             (ratio is the same across each class,
    #             samples for each class = int(np.ceil(ratio * # data in each class)) )

    if ratio <= 0 or ratio >= 1:
        raise Exception("ratio must be 0 < ratio < 1.")
    y_array = np.asarray(y)

    label_dict={}
    sample_dict=[]
    samples = []
    set_dict= set(y)
    for label in set_dict:
        lst_index=[]
        for i in range(len(y)):
            if y[i]== label:
                lst_index.append(i)
        label_dict[label]=lst_index
        num_samples= int(np.ceil(ratio * len(label_dict[label])))
        label_samples = (np.random.choice(label_dict[label], num_samples, replace))
        for sample in label_samples:
            samples.append(sample)
    return samples
