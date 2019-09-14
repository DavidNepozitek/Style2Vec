import json
import random
import numpy as np
import tensorflow as tf
import math
from pathlib import Path
import matplotlib.pyplot as plt


class SamplesGenerator:

    def __init__(self, dataset_path: str, images_path: str, neg_sample_count: int = 5, batch_size: int = 10,
                 underlying_model: str = 'inception', outfits_count_limit: int = -1, samples_count_limit: int = -1):
        """
        :param dataset_path: Path to your dataset path
        :param images_path: Path to the folder with images
        :param neg_sample_count: Number of negative samples per item
        :param batch_size: Number of samples in one batch
        :param underlying_model: Model to use: ('inception')
        :param outfits_count_limit: Maximum number of outfits used for generating samples (-1 for inf)
        :param samples_count_limit: Maximum number of samples (-1 for inf)
        """
        self.outfits_count_limit = outfits_count_limit
        self.samples_count_limit = samples_count_limit
        self.batch_size = batch_size
        self.neg_sample_count = neg_sample_count
        self.images_path = images_path
        self.dataset_path = dataset_path
        self.underlying_model = underlying_model
        self.data = None
        self.steps_per_epoch = None
        self.samples = None

        if underlying_model != 'inception':
            raise ValueError("Invalid underlying model")

    def collect_data(self):
        """
        Load data from json
        """
        with open(self.dataset_path) as json_file:
            self.data = json.load(json_file)
            if self.outfits_count_limit != -1 and self.outfits_count_limit < len(self.data):
                self.data = self.data[:self.outfits_count_limit]
        print("Dataset has been successfully loaded (" + str(len(self.data)) + " outfits).")

    def generate_samples(self):
        """
        Generates samples with labels
        :return: List of couples and labels (0/1)
        """
        if self.data is None:
            self.collect_data()

        samples = []
        for outfit in self.data:
            for item in outfit["Items"]:
                # Add positive samples
                for context in outfit["Items"]:
                    if item != context:
                        samples.append(((item, context), 1))
                # Add negative samples
                for i in range(self.neg_sample_count):
                    while True:
                        neg_outfit_index = random.randrange(len(self.data) - 1)
                        if neg_outfit_index != outfit["SetId"]:
                            break
                    neg_item = random.choice(self.data[neg_outfit_index]["Items"])
                    samples.append(((item, neg_item), 0))
        random.shuffle(samples)
        if self.samples_count_limit != -1 and self.samples_count_limit < len(samples):
            samples = samples[:self.samples_count_limit]

        self.steps_per_epoch = math.floor(len(samples) / self.batch_size)

        couples, labels = zip(*samples)
        self.samples = couples, labels
        print(str(len(samples)) + " samples generated.")

    def generate_batches(self):
        """
        Batch generator method
        """

        if self.samples is None:
            self.generate_samples()

        couples, labels = self.samples
        labels = labels
        item_target, item_context = zip(*couples)
        target_items = np.array(item_target)
        context_items = np.array(item_context)

        while True:
            current_size = 0
            batched_targets, batched_contexts, batched_labels = [], [], []
            for i, target in enumerate(target_items):
                if current_size < self.batch_size:
                    batched_targets.append(self.prep_item(target))
                    batched_contexts.append(self.prep_item(context_items[i]))
                    batched_labels.append(labels[i])
                    current_size += 1
                else:
                    yield ([np.array(batched_targets), np.array(batched_contexts)], np.array(batched_labels))
                    current_size = 0
                    batched_targets, batched_contexts, batched_labels = [], [], []
            if current_size > 0:
                yield ([np.array(batched_targets), np.array(batched_contexts)], np.array(batched_labels))

    def prep_item(self, item):
        """
        Transforms dataset item into the model input
        :param item: Dataset item
        :return: Model input
        """
        if self.underlying_model == 'inception':
            img = tf.keras.preprocessing.image.load_img(Path(self.images_path + item["ImagePath"]),
                                                        target_size=(299, 299))
            x = tf.keras.preprocessing.image.img_to_array(img)
            return tf.keras.applications.inception_v3.preprocess_input(x)
        else:
            raise ValueError("Preprocessing not defined for this model")


class Style2Vec:

    def __init__(self,
                 dataset_path: str,
                 images_path: str,
                 batch_size: int = 10,
                 epochs_count: int = 1,
                 underlying_model: str = 'inception',
                 outfits_count_limit: int = -1,
                 samples_count_limit: int = -1):
        """
        Style2Vec model wrapper
        :param dataset_path: Path to your dataset path
        :param images_path: Path to the folder with images
        :param batch_size: Number of samples in one batch
        :param epochs_count: Number of epochs
        :param underlying_model: Model to use: ('inception')
        :param outfits_count_limit: Maximum number of outfits used for generating samples (-1 for inf)
        :param samples_count_limit: Maximum number of samples to generate (-1 for inf)
        """
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

        model_context = tf.keras.applications.inception_v3.InceptionV3(
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
        for i, layer in enumerate(model_context.layers):
            layer._name = 'context_' + str(i)
            if i == len(model_context.layers) - 1:
                layer._name = 'context_last_layer'

        # Perform dot product
        dot_product = tf.keras.layers.dot(
            [self.model_target.get_layer("target_last_layer").output,
             model_context.get_layer("context_last_layer").output], axes=1)
        dot_product = tf.keras.layers.Reshape((1,))(dot_product)

        # Sigmoid layer
        output = tf.keras.layers.Dense(1, activation='sigmoid')(dot_product)

        # Create model and generator
        self.model = tf.keras.Model(inputs=[input_target, input_context], outputs=output)
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
        self.generator = SamplesGenerator(
            dataset_path,
            images_path,
            batch_size=batch_size,
            samples_count_limit=samples_count_limit,
            outfits_count_limit=outfits_count_limit
        )
        print("Style2Vec model has been successfully initialized.")

    def fit(self):
        self.generator.generate_samples()
        print("Model fitting has started.")
        self.history = self.model.fit_generator(
            self.generator.generate_batches(),
            steps_per_epoch=self.generator.steps_per_epoch,
            epochs=self.epochs_count,
            verbose=2
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

    def plot_history(self):
        plt.plot(self.history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
        plt.savefig('plot.png')

    def eval(self, dataset_path: str, outfits_count_limit: int):
        self.model.evaluate()


class Embedding:
    model = ...  # type: Model

    def __init__(self, model: tf.keras.models.Model):
        self.model = model

    def get_embedding(self, items):
        embedding = []
        paths = []

        for item in items:
            try:
                prep = self.prep_item(item)
            except:
                continue
            batch = np.array([prep])
            features = self.model.predict_on_batch(batch)
            embedding.append(features)
            paths.append(item)

        return embedding, paths

    def get_features(self, path_to_img):
        preprocessed = self.prep_item(path_to_img)
        features = self.model.predict(preprocessed)
        return features

    def collect_data(self, dataset_path, items_count_limit=-1):
        """
        Load data from json
        """
        with open(dataset_path) as json_file:
            data = json.load(json_file)
            result = []
            for outfit in data:
                for item in outfit['Items']:
                    if items_count_limit != -1 and len(result) >= items_count_limit:
                        print("Dataset has been successfully loaded (" + str(len(result)) + " items).")
                        return result
                    result.append(item["ImagePath"])

        print("Dataset has been successfully loaded (" + str(len(result)) + " items).")
        return result

    @staticmethod
    def prep_item(path_to_img: str):
        """
        Transforms dataset item into the model input
        :param path_to_img:
        :param item: Dataset item
        :return: Model input
        """
        img = tf.keras.preprocessing.image.load_img(
            Path("../dataset/images/" + path_to_img), target_size=(299, 299))
        x = tf.keras.preprocessing.image.img_to_array(img)
        return tf.keras.applications.inception_v3.preprocess_input(x)


model = Style2Vec("train_no_dup_out.json", "../dataset/images/", batch_size=1, outfits_count_limit=3)
target = model.model_target  # type: tf.keras.models.Model
# target.load_weights('../models/beta14.h5', True)

embedding = Embedding(target)
data = embedding.collect_data("valid_no_dup_out.json")[:3000]
emb, paths = embedding.get_embedding(data)

emb_array = np.array(emb)
paths_array = np.array(paths)
np.save("embeding_imagenet", emb_array)
np.save("paths_imagenet", paths_array)