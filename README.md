# Style2Vec

Implementation of fashion products vector representation based on [Style2Vec paper](https://arxiv.org/abs/1708.04014).
> Based on the intuition of distributional semantics used in word embeddings, Style2Vec learns the representation of a fashion item using other items in matching outfits as context. Two different convolutional neural networks are trained to maximize the probability of item co-occurrences.

TODO: Add evaluation info


## Data Sources
- [Polyvore Dataset](https://github.com/xthan/polyvore-dataset) was used for model training.
    - The dataset was cleaned of non-wearable categories such as furniture, make-up etc. (The cleaned version can be found in `data/processed` directory.
    - Training partition contains 93 963 items in 16 920 outfits
    - Validation partition contains 6969 items in 1309 outfits
- [DeepFashion: Attribute Prediction](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html) was used for model evaluation


## How to run experiments



### Requirements
- Python 3.5.4
- Tensorflow
- Matplotlib

## Results


## Project Organization


    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Project documentation
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── style2vec          <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── experiments       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations