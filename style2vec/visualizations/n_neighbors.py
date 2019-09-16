import numpy as np
import sklearn.neighbors as neighbors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from pathlib import Path


def nearest_neighbors(embedding_path, samples_count, neighbors_count):
    embedding = np.load(embedding_path)
    embedding = np.squeeze(embedding, axis=1)
    nn = neighbors.NearestNeighbors(metric='cosine', n_neighbors=neighbors_count).fit(embedding)

    chosen_samples = []
    for i in range(samples_count):
        chosen_samples.append(random.choice(embedding))

    return nn.kneighbors(np.array(chosen_samples))


# em = np.load('../../data/processed/df_embedding.npy')
# imgs = np.load('../../data/processed/df_paths.npy')
#
# def plot_figures(figures, nrows = 5, ncols=10):
#     """Plot a dictionary of figures.
#
#     Parameters
#     ----------
#     figures : <title, figure> dictionary
#     ncols : number of columns of subplots wanted in the display
#     nrows : number of rows of subplots wanted in the figure
#     """
#
#     fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(30, 30), dpi=80)
#     for ind,path in enumerate(figures):
#         img = mpimg.imread(Path("../../../dataset/" + imgs[path]))
#         axeslist.ravel()[ind].imshow(img)
#         axeslist.ravel()[ind].set_title(path)
#         axeslist.ravel()[ind].set_axis_off()
#     plt.tight_layout()  # optional


# em = np.squeeze(em, axis=1)
#
# nr = neighbors.NearestNeighbors(metric='cosine', n_neighbors=10).fit(em)
#
# chosen = []
#
# for i in range(15):
#     chosen.append(random.choice(em))
#
# dist, kn = nr.kneighbors(np.array(chosen))
#
# plot_figures(kn.flatten(), 15)
# r = random.randint(100, 999)
# plt.savefig("../../results/nr" + str(r) + ".png")
# plt.show()

