import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from style2vec.models.style2vec import Style2Vec


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


model = Style2Vec("train_no_dup_out.json", "../data/images/", batch_size=1, outfits_count_limit=3)
target = model.model_target  # type: tf.keras.models.Model
target.load_weights('logs/20190910-193051/model20190910-193051.h5', True)
embedding = Embedding(target)
data = embedding.collect_data("../data/label/valid_no_dup_out.json")
emb, paths = embedding.get_embedding(data)

emb_array = np.array(emb)
paths_array = np.array(paths)
np.save("logs/20190910-193051/embeding_m193051", emb_array)
np.save("logs/20190910-193051/paths_m193051", paths_array)
