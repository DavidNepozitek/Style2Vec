import argparse
import matplotlib.pyplot as plt
import numpy as np
import datetime
from functools import reduce
from style2vec.data import deepfashion_prep as preprocessing
from style2vec.visualizations.n_neighbors import nearest_neighbors


def main():
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
    # dist, n = nearest_neighbors(args.emb_path, 4000, 500)

    n = []
    for i in range(4000):
        n.append(np.random.choice(4000, 500, False))

    get_stats(n, paths, items, args.attr_types_path)


def get_stats(n, paths, items, attr_types_path):
    type_mask = preprocessing.get_attr_type_mask(attr_types_path, 5)

    plt.figure(figsize=(20, 20))
    w_averages = []
    for i, neighbors in enumerate(n):
        neighbors = n[i]
        item = items[paths[neighbors[0]]]
        attr_count = count_attributes(item)
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
        # plt.plot(range(len(means)), matches, label=str(attr_count))
        if sum(matches) == 0:
            continue
        avg = np.average(range(len(matches)), weights=matches)
        w_averages.append(avg)
    print(np.median(w_averages))
    print(np.mean(w_averages))
    # plt.savefig("../../results/fig" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png")
    # plt.show()

def compare_attributes(item, neighbor, type_mask=None):
    count = 0
    for i, attr in enumerate(item.attributes):
        if type_mask is not None and i not in type_mask:
            continue
        if attr == "1" and attr == neighbor.attributes[i]:
            count += 1

    return count


def count_attributes(item):
    return reduce(lambda current_sum, value: value == current_sum + 1 if value == "1" else current_sum, item.attributes, 0)


main()
