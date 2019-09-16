# Style2Vec Report

David Nepožitek

---

## Problem Definition
The goal of this experiment is to implement a model for fashion products vector representation based on [Style2Vec](https://arxiv.org/abs/1708.04014) paper. The final model should be able to create fashion items embedding, where items with similiar style are near to one another in the latent space.

## Motivation

## Data Description
[Polyvore Dataset](https://github.com/xthan/polyvore-dataset) was used for training the model. It is made out of outfits obtained from polyvore.com, a former website allowing users to create outfits out of various fashion products. The original dataset contains 21,889 outfits with on average 6.5 fashion items. There is a image of each item, usually on white background, and some additional information, e. g. number of likes, price, and so on. To improve the representation we cleaned the dataset of items from 191 non-wearable categories such as furniture, make-up, etc. Then we removed outfits with less than 3 items. So the final dataset contains __XX,XXX__ outfits and the average number of items is __X.X__. 

For model evaluation [DeepFashion: Attribute Prediction Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html) was used. It contains 289,222 images of fashion products, some of them on white background but most of them worn by models. That is why bounding boxes are provided to eliminate visual noise of the background. There are 1,000 item attributes divided into 5 attribute types. We used the validation partition of this dataset containing 4,000 items.

## Experiment Description
The approach is inspired by Word2Vec model that is based on word coocurrence in a context window. In our model a fashion item is treated as a word and outfits corresponds to context windows. The model uses two convolution neural networks. One of the network is used as a projection to target item embedding vector and the other is used for projection context items to weight vectors. Then dot product is computed between target item embedding vector and each context item in the same dataset, the ouput is given to softmax function. The goal is to maximize the probability of cooccurence of items in same outfits. More details can be found in the [original paper](https://arxiv.org/abs/1708.04014).

## Implementation
The model is constructed with Keras Functional API. Two InceptionV3 networks are used with initial weights from ImageNet classification task. Only top layers of these networks are then trained. The output of the networks is passed to dot product from which a sigmoid function is computed. We don't compute the softmax function since it's expensive, we use negative sampling instead. For each item a few items from other outfits are chosen as negative samples. Binary cross-entropy loss function is used because we want the model ouput of positive samples to be 1 and of negative samples to be 0. The final model was trained with minibatch gradient descent with batch size of __XX__, adam optimizer with default parameters for __XX__ epochs.

## Evaluation


## Results


## Conclusion