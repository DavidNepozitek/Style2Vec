import tensorflow as tf
import numpy as np
import re
from PIL import Image as PILImage
from tensorflow.python.keras.preprocessing import image


class Item:

    def __init__(self, path):
        self.attributes = None  # type: list
        self.bbox = None  # type: tuple
        self.path = path  # type: str


def parse(path_to_attributes, path_to_bboxes, path_to_partition, partition, load_attributes=False):
    result = {}

    partition_file = open(path_to_partition, "r", encoding="utf-8")
    lines_count = int(partition_file.readline())
    partition_file.readline()  # Header

    # Load selected partition
    for i in range(lines_count):
        item_data = partition_file.readline().split()
        if item_data[1] == partition:
            path = item_data[0]
            result.update({path: Item(path)})

    bboxes_file = open(path_to_bboxes, encoding="utf-8")
    lines_count = int(bboxes_file.readline())
    bboxes_file.readline()  # Header

    # Load bounding boxes
    for i in range(lines_count):
        item_data = bboxes_file.readline().split()
        path = item_data[0]
        box = tuple(list(map(int, item_data[1:])))
        if path in result:
            result.get(path).bbox = box

    if not load_attributes:
        return result

    attribute_file = open(path_to_attributes, "r", encoding="utf-8")
    lines_count = int(attribute_file.readline())
    attribute_file.readline()  # Header

    # Load attributes
    for i in range(lines_count):
        item_data = attribute_file.readline().split()
        path = item_data[0]
        if path in result:
            result[path].attributes = item_data[1:]

    return result


def get_attr_type_mask(attr_types_path, type_id):
    mask = []
    attribute_file = open(attr_types_path, "r", encoding="utf-8")
    lines_count = int(attribute_file.readline())
    attribute_file.readline()  # Header
    # Load attributes types
    for i in range(lines_count):
        attribute_data = re.split(r'\s\s+', attribute_file.readline(), 2)
        if attribute_data[1].rstrip() == str(type_id):
            mask.append(i)
    return mask


def prep_image(item: Item, data_dir: str) -> np.ndarray:
    img = image.load_img(data_dir + item.path)  # type: PILImage.Image
    img = img.crop(item.bbox)
    img = img.resize((299, 299))
    img_array = image.img_to_array(img)
    return tf.keras.applications.inception_v3.preprocess_input(img_array)


def get_embedding(model, items: dict, data_dir: str):
    embedding = []
    paths = []

    for item in items.values():  # type: Item
        try:
            prep = prep_image(item, data_dir)
        except Exception as e:
            print(e)
            continue
        batch = np.array([prep])
        features = model.predict_on_batch(batch)
        embedding.append(features)
        paths.append(item.path)

    return embedding, paths
