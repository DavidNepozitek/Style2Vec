import os
import argparse
import tensorflow as tf
import numpy as np
from style2vec.data import deepfashion_prep as preprocessing
from style2vec.models.style2vec import Style2Vec


def main():
    """
    Create Deep Fashion dataset embedding
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--attr-path", type=str, help="Path to item attributes data")
    parser.add_argument("--bbox-path", type=str, help="Path to item bounding boxes")
    parser.add_argument("--part-path", type=str, help="Path to dataset partition file")
    parser.add_argument("--model-path", type=str, help="Path to target model weights")
    parser.add_argument("--img-base-dir", type=str, help="Path where 'img' dir is located")
    args = parser.parse_args()
    items = preprocessing.parse(args.attr_path, args.bbox_path, args.part_path, "val")
    print(str(len(items)) + " items parsed")

    model = Style2Vec("train_no_dup_out.json", "../data/images/", batch_size=1, outfits_count_limit=3)
    target = model.model_target  # type: tf.keras.models.Model
    target.load_weights(args.model_path, True)

    emb, paths = preprocessing.get_embedding(target, items, args.img_base_dir)

    print("Created embedding from " + str(len(emb)) + " items")

    emb_array = np.array(emb)
    paths_array = np.array(paths)
    print(emb_array.shape)
    np.save(os.path.dirname(os.path.realpath(__file__)) + "/../../data/processed/df_embedding_rel", emb_array)
    np.save(os.path.dirname(os.path.realpath(__file__)) + "/../../data/processed/df_paths_rel", paths_array)
    print("Embedding created")

main()
