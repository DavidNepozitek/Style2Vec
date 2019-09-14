import numpy as np
import sklearn.neighbors as neighbors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from pathlib import Path

em = np.load('../models/embeding20190910-193051.npy')
imgs = np.load('../models/paths20190910-193051.npy')

def plot_figures(figures, nrows = 5, ncols=10):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,figsize=(30, 30), dpi=80)
    for ind,path in enumerate(figures):
        #img = mpimg.imread("../dataset/images/" + imgs[path])
        axeslist.ravel()[ind].imshow(mpimg.imread(Path("../dataset/images/" + imgs[path])))
        axeslist.ravel()[ind].set_title(path)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional


em = np.squeeze(em, axis=1)

nr = neighbors.NearestNeighbors(metric='cosine', n_neighbors=10).fit(em)

chosen = []

for i in range(15):
    chosen.append(random.choice(em))

dist, kn = nr.kneighbors(np.array(chosen))

plot_figures(kn.flatten(), 15)
r = random.randint(100, 999)
plt.savefig("nr" + str(r) + ".png")
plt.show()

