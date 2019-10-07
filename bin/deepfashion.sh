#!/usr/bin/env bash

MODEL_NAME="model20190910-192827.h5"
cd ..

# Create representation
#python -m "style2vec.features.df_embedding" \
#    --attr-path "data/raw/list_attr_img.txt" \
#    --bbox-path "data/raw/list_bbox.txt" \
#    --part-path "data/raw/list_eval_partition.txt" \
#    --model-path "models/${MODEL_NAME}" \
#    --img-base-dir "data/raw/"

python -m "style2vec.features.df_neighbors" \
    --attr-types-path "data/raw/list_attr_cloth.txt" \
    --part-path "data/raw/list_eval_partition.txt" \
    --attr-path "data/raw/list_attr_img.txt" \
    --bbox-path "data/raw/list_bbox.txt" \
    --emb-path="data/processed/df_embedding.npy" \
    --emb-path-in="data/processed/df_embedding_in.npy" \
    --paths-path="data/processed/df_paths.npy"