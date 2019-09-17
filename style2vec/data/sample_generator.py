import json
import random
import numpy as np
import tensorflow as tf
import math
from pathlib import Path


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
