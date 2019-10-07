# Style2Vec

Implementation of fashion products vector representation based on [Style2Vec: Representation Learning for Fashion Items from Style Sets](https://arxiv.org/abs/1708.04014).
> Based on the intuition of distributional semantics used in word embeddings, Style2Vec learns the representation of a fashion item using other items in matching outfits as context. Two different convolutional neural networks are trained to maximize the probability of item co-occurrences.

## Docs
- [Report](docs/Report.md)

## Data Sources
- [Polyvore Dataset](https://github.com/xthan/polyvore-dataset) was used for model training.
- [DeepFashion: Attribute Prediction](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html) was used for model evaluation

### Requirements
- Tensorflow > 1.14
- Matplotlib
- scikit-learn

## Results
Sequence of nearest neighbors from Style2Vec embedding on each line:
![](figures/nnstyle2vec.png)

## Project Organization


    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Project documentation and reports
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    |
    └── style2vec          <- Source code for use in this project.
        ├── data           <- Scripts to process data
        │   │
        │   ├── deepfashion_prep.py <- DF dataset preprocessing
        |   ├── sample_generator.py <- Samples generator for model training
        │   └── preprocessing.py <- Image preprocessing
        │
        ├── features    <- Scripts to turn raw data into features for modeling
        |   ├── df_embedding.py <- Deepfashion dataset embedding
        │   └── polyvore_embedding.py <- Polyvore dataset embedding
        │
        ├── models         <- Scripts to train models and then use trained models to make predictions
        │   └── style2vec.py <- Style2Vec model training
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            │
            ├── df_attr_comparison.py <- Visualization of nearest negihbors with their attributes
            ├── df_exploration.py <- Deep Fashion dataset statistical exploration
            ├── df_neighbors.py <- Model validation
            ├── n_neighbors.py <- sk-learn n-neighbors wrapper
            └── polyvore_neighbord.py <- Nearest neighbors visualization of Polyvore dataset embedding