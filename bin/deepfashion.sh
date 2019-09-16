#!/usr/bin/env bash

MODEL_NAME="model20190910-192827.h5"
cd ..

# Create representation
python -m "style2vec.experiments.neighbors_style" \
    --attr-path "data/raw/list_attr_img.txt" \
    --bbox-path "data/raw/list_bbox.txt" \
    --part-path "data/raw/list_eval_partition.txt" \
    --model-path "models/${MODEL_NAME}" \
    --img-base-dir "data/raw/"


#python -m "style2vec.visualizations.n_neighbors"