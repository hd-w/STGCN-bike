import os
import zipfile
import numpy as np
import torch

def min_max_norm(x,min,max):
    return (x-min)/(max-min)

def calculate_neiborhoods(position_LA, position_LO):
    length = len(position_LA)
    neiborhood_lists = []
    for i in range(length):
        neiborhoods = []
        for j in range(length):
            if i == j:
                continue
            else:
                distance = ((position_LA[i]-position_LA[j])**2+ \
                (position_LO[i]-position_LO[j])**2)**0.5
                # print(distance)
                if distance < 0.005:
                    neiborhoods.append(j)
        neiborhood_lists.append(neiborhoods)
    return neiborhood_lists



def load_metr_la_data():
    A = np.load("./data/adjacency_matrix_2.npy")
    X = np.load("./data/history_node_all_features_times_15_average.npy").transpose((2, 1, 0))
    # X = X[:,[0,1,2,3],:]
    # X = np.expand_dims(X, axis=1)
    # print(X.shape)
    X = X.astype(np.float64)  # (110,8,8822)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds

def load_2_metr_la_data():
    A = np.load("data/adjacency_matrix.npy")
    A_2 = np.load("data/adjacency_matrix_cov.npy")
    X = np.load("data/history_node_all_features_times_15_average.npy").transpose((2, 1, 0))
    # X = X[:,[0,1,2,3],:]
    # X = np.expand_dims(X, axis=1)
    # print(X.shape)
    X = X.astype(np.float64)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, A_2, X, means, stds

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float64))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    aa = diag.reshape((-1, 1))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j]) # X[:,0,:] is available bike

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))

def inference_data_transform(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))