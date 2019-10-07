import argparse
import pandas as pd
from style2vec.data import deepfashion_prep as preprocessing
import os


def main():
    """
    Deep Fashion dataset exploration
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--attr-path", type=str, help="Path to item attributes data")
    parser.add_argument("--bbox-path", type=str, help="Path to item bounding boxes")
    parser.add_argument("--part-path", type=str, help="Path to dataset partition file")
    parser.add_argument("--emb-path", type=str, help="Path to dataset embedding")
    parser.add_argument("--attr-types-path", type=str, help="Path to dataset embedding")
    parser.add_argument("--paths-path", type=str, help="Path to a path file created during embedding creation")
    args = parser.parse_args()

    part = "val"
    df_path = os.path.dirname(os.path.realpath(__file__)) + "/../../data/processed/attr_dataframe_" + part
    attrs = preprocessing.get_attr_types(args.attr_types_path)
    if os.path.exists(df_path):
        attr_dataframe = pd.read_pickle(df_path)
    else:
        items = preprocessing.parse(args.attr_path, args.bbox_path, args.part_path, part, True)
        attrs_data = []

        for item in items.values():  # type: preprocessing.Item
            mapped = map(lambda x: 1 if x == "1" else 0, item.attributes)
            attrs_data.append(mapped)
        attr_dataframe = pd.DataFrame.from_records(attrs_data)
        attr_dataframe.to_pickle(df_path)

    sum0 = attr_dataframe.sum()  # type: pd.Series
    sum1 = attr_dataframe.sum(axis=1)  # type: pd.Series
    print(sum0.describe())
    print(sum0)
    print(sum0.mean())
    print(sum0.median())
    print(sum1.describe())
    print(sum1)
    print(sum1.mean())
    print(sum1.median())
    sorted_counts = sum0.sort_values(ascending=False)

    treshold = 500
    above_tresh = []
    for index, value in sorted_counts.items():
        print(attrs[index][1] + " " + attrs[index][0] + ": " + str(value))
        if value >= treshold:
            above_tresh.append(index)

    print(above_tresh)


main()
