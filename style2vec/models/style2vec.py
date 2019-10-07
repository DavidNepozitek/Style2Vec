import datetime
import json
import random
import numpy as np
import tensorflow as tf
import math
import os
from pathlib import Path
from tensorboard.plugins.hparams import api as hp
from style2vec.data.sample_generator import SamplesGenerator
import time

date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
dir_path = os.path.dirname(os.path.realpath(__file__))


class Style2Vec:

    def __init__(self,
                 dataset_path: str,
                 images_path: str,
                 batch_size: int = 10,
                 epochs_count: int = 1,
                 underlying_model: str = 'inception',
                 outfits_count_limit: int = -1,
                 samples_count_limit: int = -1,
                 hparams=None):
        """
        Style2Vec model wrapper
        :type hparams: Hyperparameters
        :param dataset_path: Path to your dataset path
        :param images_path: Path to the folder with images
        :param batch_size: Number of samples in one batch
        :param epochs_count: Number of epochs
        :param underlying_model: Model to use: ('inception')
        :param outfits_count_limit: Maximum number of outfits used for generating samples (-1 for inf)
        :param samples_count_limit: Maximum number of samples to generate (-1 for inf)
        """
        self.hparams = hparams
        self.epochs_count = epochs_count
        self.history = None
        # Create input layers
        input_target = tf.keras.layers.Input((299, 299, 3))
        input_context = tf.keras.layers.Input((299, 299, 3))

        # Initialize underlying models
        self.model_target = tf.keras.applications.inception_v3.InceptionV3(  # type: tf.keras.models.Model
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_tensor=input_target
        )

        self.model_context = tf.keras.applications.inception_v3.InceptionV3(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_tensor=input_context
        )

        # Rename layers
        for i, layer in enumerate(self.model_target.layers):
            layer._name = 'target_' + str(i)
            if i == len(self.model_target.layers) - 1:
                layer._name = 'target_last_layer'
        for i, layer in enumerate(self.model_context.layers):
            layer._name = 'context_' + str(i)
            if i == len(self.model_context.layers) - 1:
                layer._name = 'context_last_layer'

        if hparams[HP_FINE_TUNE]:
            # Set up fine-tuning
            for layer in self.model_target.layers[:249]:
                layer.trainable = False
            for layer in self.model_target.layers[249:]:
                layer.trainable = True
            for layer in self.model_context.layers[:249]:
                layer.trainable = False
            for layer in self.model_context.layers[249:]:
                layer.trainable = True

        # Perform dot product
        dot_product = tf.keras.layers.dot(
            [self.model_target.get_layer("target_last_layer").output,
             self.model_context.get_layer("context_last_layer").output], axes=1)
        dot_product = tf.keras.layers.Reshape((1,))(dot_product)

        # Sigmoid layer
        output = tf.keras.layers.Dense(1, activation='sigmoid')(dot_product)

        # Create model and generator

        self.model = tf.keras.Model(inputs=[input_target, input_context], outputs=output)
        self.model = tf.keras.utils.multi_gpu_model(self.model, gpus=2)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.generator = SamplesGenerator(
            dataset_path,
            images_path,
            batch_size=batch_size,
            samples_count_limit=samples_count_limit,
            outfits_count_limit=outfits_count_limit
        )
        print("Style2Vec model has been successfully initialized.")

    def fit(self):
        print("Model fitting has started.")
        self.generator.generate_samples()
        self.history = self.model.fit_generator(
            self.generator.generate_batches(),
            steps_per_epoch=self.generator.steps_per_epoch,
            epochs=self.epochs_count,
            verbose=2,
            callbacks=[
                tf.keras.callbacks.TensorBoard("logs/" + date,
                                               update_freq=10000, write_graph=False),  # log metrics
                tf.keras.callbacks.ModelCheckpoint("logs/" + date + "/chckpts/weights.{epoch:02d}.hdf5")
            ],
            workers=8,
            use_multiprocessing=True,
        )

    def plot_model(self):
        tf.keras.utils.plot_model(
            self.model,
            to_file='model.png',
            show_shapes=False,
            show_layer_names=True,
            rankdir='TB'
        )

    def save_weights(self, filepath: str):
        self.model.save_weights(filepath)

    def save(self, model_filepath: str = 'model.h5'):
        self.model.save(model_filepath)


HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([8, 16, 24, 32, 48]))
HP_NEGATIVE_SAMPLES = hp.HParam('negative_samples', hp.Discrete([6, 12, 18]))
HP_FINE_TUNE = hp.HParam('fine_tune', hp.Discrete([True, False]))

METRIC_ACCURACY = 'accuracy'

hyperparams = {
    HP_BATCH_SIZE: 5,
    HP_NEGATIVE_SAMPLES: 6,
    HP_OPTIMIZER: 'adam',
    HP_FINE_TUNE: True
}

model = Style2Vec("../data/label/train_no_dup_out.json",
                  "../data/images/",
                  hyperparams[HP_BATCH_SIZE],
                  outfits_count_limit=50,
                  epochs_count=5,
                  hparams=hyperparams)

try:
    start = time.clock()
    model.fit()
    end = time.clock()
    model.save('logs/' + date + '/' + 'model' + date + '.h5')
    with open('logs/' + date + '/meta.txt', "w+") as time_file:
        time_file.write('batch 24, limit -1, adam, e 5, fine tune, neg 6\n')
        time_file.write(str(start - end))
    print("Succesfully finished.")
except Exception as e:
    model.save('model' + date + '_err.h5')
    with open('err' + date + '.txt', "w+") as err_file:
        err_file.write(str(e))
    print("Exited with error.")
