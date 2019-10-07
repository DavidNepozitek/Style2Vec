import numpy as np
import sklearn.neighbors as neighbors
import random


def nearest_neighbors(embedding_path, samples_count, neighbors_count):
    """
    Find nearest neighbors of random samples
    :param embedding_path: Path to embedding
    :param samples_count: Number of samples
    :param neighbors_count: Number of neighbors to find
    :return: (distances, ordered neighbors)
    """
    embedding = np.load(embedding_path)
    embedding = np.squeeze(embedding, axis=1)
    nn = neighbors.NearestNeighbors(metric='cosine', n_neighbors=neighbors_count, n_jobs=-1).fit(embedding)

    chosen_samples = []
    for i in range(samples_count):
        chosen_samples.append(random.choice(embedding))

    return nn.kneighbors(np.array(chosen_samples))


def fixed_nearest_neighbors(embedding_path, indices_list, neighbors_count):
    """
    Find nearest neighbors for given samples
    :param embedding_path: Path to embedding
    :param indices_list: Indices of samples
    :param neighbors_count: Number of neighbors to find
    :return: (distances, ordered neighbors)
    """
    embedding = np.load(embedding_path)
    embedding = np.squeeze(embedding, axis=1)
    nn = neighbors.NearestNeighbors(metric='cosine', n_neighbors=neighbors_count, n_jobs=-1).fit(embedding)

    chosen_samples = []
    for i in indices_list:
        chosen_samples.append(embedding[i])

    return nn.kneighbors(np.array(chosen_samples))
