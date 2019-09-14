#!/usr/bin/env bash

MODEL_NAME="model20190910-192827.h5"
cd ..
python -m "style2vec.experiments.neighbors_style" \
    --attr-path "data/raw/list_attr_img.txt" \
    --bbox-path "data/raw/list_bbox.txt" \
    --model-path "models/${MODEL_NAME}" \
    --img-base-dir "data/raw/"
