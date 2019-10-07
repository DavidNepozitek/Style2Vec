from style2vec.visualizations.n_neighbors import nearest_neighbors
import style2vec.data.deepfashion_prep as preprocessing
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import datetime
import argparse
from pathlib import Path


def main():
    """
    Render nearest neighbors for random samples and show their attributes
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--attr-path", type=str, help="Path to item attributes data")
    parser.add_argument("--bbox-path", type=str, help="Path to item bounding boxes")
    parser.add_argument("--part-path", type=str, help="Path to dataset partition file")
    parser.add_argument("--emb-path", type=str, help="Path to dataset embedding")
    parser.add_argument("--attr-types-path", type=str, help="Path to dataset embedding")
    parser.add_argument("--paths-path", type=str, help="Path to a path file created during embedding creation")

    args = parser.parse_args()
    paths = np.load(args.paths_path)
    items = preprocessing.parse(args.attr_path, args.bbox_path, args.part_path, "val", True)
    attrs = preprocessing.parse_attribute_names(args.attr_types_path)

    dist, kn = nearest_neighbors('../../data/processed/df_embedding_rel.npy', 10, 15)
    plot_figures(kn.flatten(), paths, attrs, items)

    plt.savefig("../../results/nn-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png")
    plt.show()


def plot_figures(neighbors, paths, attrs, items, rows=10, cols=15):
    """
    Plot nearest neighbors and attribute names
    :param neighbors: List of nearest neighbors
    :param paths: List of paths to product images
    :param attrs: Attribute names mapping
    :param items: List of parsed items
    :param rows: Number of rows
    :param cols: Number of columns
    """

    # Create subplots with 2 * number of rows
    fig, axeslist = plt.subplots(ncols=cols, nrows=rows * 2, figsize=(45, 45), dpi=72)
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    for i in range(0, rows * 2 - 1, 2):
        for j in range(cols):
            # Show product image
            path = paths[neighbors[(i // 2) * cols + j]]
            img = mpimg.imread(Path("../../../dataset/" + path))
            axeslist[i][j].imshow(img)
            axeslist[i][j].set_axis_off()
            attr_names = preprocessing.get_attribute_names(items[path], attrs)
            # Render attributes below
            names = '\n'.join(attr_names)
            axeslist[i + 1][j].text(0, 1, names, verticalalignment='top', transform=axeslist[i + 1][j].transAxes)
            axeslist[i + 1][j].set_axis_off()


main()
