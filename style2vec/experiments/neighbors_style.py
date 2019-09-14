import argparse
import tensorflow as tf
import numpy as np
from style2vec.data import deepfashion_prep as preprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attr-path", type=str, help="Path to item attributes data")
    parser.add_argument("--bbox-path", type=str, help="Path to item bounding boxes")
    parser.add_argument("--model-path", type=str, help="Path to target model weights")
    parser.add_argument("--img-base-dir", type=str, help="Path where 'img' dir is located")
    args = parser.parse_args()
    print("args parsed")
    items = preprocessing.parse(args.attr_path, args.bbox_path)
    print("items parsed")
    input_target = tf.keras.layers.Input((299, 299, 3))
    target = tf.keras.applications.inception_v3.InceptionV3(  # type: tf.keras.models.Model
        include_top=False,
        pooling='avg',
        input_tensor=input_target
    )

    # Rename layers
    for i, layer in enumerate(target.layers):
        layer._name = 'target_' + str(i)
        if i == len(target.layers) - 1:
            layer._name = 'target_last_layer'

    print("model created")
    target.load_weights(args.model_path, True)
    print("Weights loaded")

    emb, paths = preprocessing.get_embedding(target, items, args.img_base_dir)

    emb_array = np.array(emb)
    paths_array = np.array(paths)
    np.save(__file__ + "../../data/processed/df_embedding", emb_array)
    np.save(__file__ + "../../data/processed/df_paths", paths_array)
    print("Embedding created")


main()
