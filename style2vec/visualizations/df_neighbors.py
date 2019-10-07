import argparse
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
from style2vec.data import deepfashion_prep as preprocessing
from style2vec.visualizations.n_neighbors import fixed_nearest_neighbors
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attr-path", type=str, help="Path to item attributes data")
    parser.add_argument("--bbox-path", type=str, help="Path to item bounding boxes")
    parser.add_argument("--part-path", type=str, help="Path to dataset partition file")
    parser.add_argument("--emb-path", type=str, help="Path to dataset embedding")
    parser.add_argument("--emb-path-in", type=str, help="Path to dataset embedding")
    parser.add_argument("--attr-types-path", type=str, help="Path to dataset embedding")
    parser.add_argument("--paths-path", type=str, help="Path to a path file created during embedding creation")
    args = parser.parse_args()
    paths = np.load(args.paths_path)
    items = preprocessing.parse(args.attr_path, args.bbox_path, args.part_path, "val", True)

    # n = []
    # for i in range(4000):
    #     n.append(np.random.choice([x for x in range(4000) if x != i], 500, False))

    plt.figure(figsize=(20, 20))
    chosen = np.random.choice(len(paths), 5000, replace=False)
    dist, n = fixed_nearest_neighbors(args.emb_path_in, chosen, 20)
    get_stats(n, paths, items, args.attr_types_path, imagenet=True)
    dist, n = fixed_nearest_neighbors(args.emb_path, chosen, 20)
    get_stats(n, paths, items, args.attr_types_path, imagenet=False)
    show()
    # dist, n = nearest_neighbors(args.emb_path, 20000, 25)
    # get_stats(n, paths, items, args.attr_types_path)
    # dist, n = nearest_neighbors(args.emb_path, 5000, 200)
    # get_stats(n, paths, items, args.attr_types_path)
    # dist, n = nearest_neighbors(args.emb_path, 1000, 1000)
    # get_stats(n, paths, items, args.attr_types_path)


def get_stats(n, paths, items, attr_types_path, imagenet=False):
    if imagenet:
        colors = ['lightskyblue', 'blue']
    else:
        colors = ['lightcoral', 'red']

    type_mask = preprocessing.get_attr_type_mask(attr_types_path, [1, 2, 5])

    w_averages = []
    global_matches = []

    for i, neighbors in enumerate(n):
        neighbors = n[i]
        item = items[paths[neighbors[0]]]
        means = []
        sums = []
        matches = []
        last_sum = 0
        for j, neighbor in enumerate(neighbors[1:]):
            match_count = compare_attributes(item, items[paths[neighbor]], type_mask)
            matches.append(match_count)
            means.append((last_sum + match_count) / (j + 1))
            last_sum = last_sum + match_count
            sums.append(last_sum)
        global_matches.append(np.array(matches))
        if sum(matches) == 0:
            continue
        avg = np.average(range(len(matches)), weights=matches)
        w_averages.append(avg)

    mean = np.mean(global_matches, axis=0)
    print(np.median(w_averages))
    print(np.mean(w_averages))

    x_plot = range(len(mean))
    X_plot = np.array(x_plot)[:, np.newaxis]

    plt.plot(x_plot, mean, color=colors[0])

    model = make_pipeline(PolynomialFeatures(3), Ridge())
    model.fit(X_plot, mean)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, color=colors[1], label="ImageNet" if imagenet else "Style2Vec", linewidth=3, zorder=5)


def show():
    plt.legend(loc='upper right')
    filepath = os.path.dirname(os.path.realpath(__file__)) + "/../../results/fig" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + ".png"
    print(filepath)
    plt.savefig(filepath)
    plt.show()


def compare_attributes(item, neighbor, type_mask=None):
    union = 0
    intersection = 0
    if type_mask is None:
        type_mask = range(len(item.attributes))

    for i in type_mask:
        if item.attributes[i] == "1" and neighbor.attributes[i] == "1":
            intersection += 1
        if item.attributes[i] == "1" or neighbor.attributes[i] == "1":
            union += 1

    if union == 0:
        return 1

    return intersection / union


def count_attributes(item, type_mask=None):
    count = 0
    if type_mask is None:
        type_mask = range(len(item.attributes))

    for i in type_mask:
        if item.attributes[i] == "1":
            count += 1

    return count


main()
