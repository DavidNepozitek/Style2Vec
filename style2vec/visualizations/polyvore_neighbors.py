import numpy as np
import sklearn.neighbors as neighbors
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pathlib import Path

em = np.load('../../models/embeding_in.npy')
imgs = np.load('../../models/paths20190910-193051.npy')


def plot_figures(figures, nrows=15, ncols=15):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(30, 30), dpi=72)

    for ind,path in enumerate(figures):
        img = mpimg.imread(Path("../../../dataset/images/" + imgs[path]))
        axeslist.ravel()[ind].imshow(img)
        axeslist.ravel()[ind].set_axis_off()
    # plt.tight_layout()  # optional


em = np.squeeze(em, axis=1)

nr = neighbors.NearestNeighbors(metric='cosine', n_neighbors=15).fit(em)

fixed = [2006, 2755, 4831, 1550, 5396, 560, 4183, 1265, 23, 3122, 2850, 2314, 2599, 3598, 125]
chosen = []

for i in fixed:
    chosen.append(em[i])

# for i in range(15):
#     chosen.append(random.choice(em))

dist, kn = nr.kneighbors(np.array(chosen))

plot_figures(kn.flatten(), 15)
r = random.randint(100, 999)
plt.savefig("../../results/nr" + str(r) + ".png")

plt.show()
