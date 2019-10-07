import tensorflow as tf
import numpy as np
import re
from PIL import Image as PILImage
from tensorflow.python.keras.preprocessing import image


class Item:
    """Parsed DeepFashion dataset item"""

    def __init__(self, path):
        self.attributes = None  # type: list
        self.bbox = None  # type: tuple
        self.path = path  # type: str


def parse(path_to_attributes, path_to_bboxes, path_to_partition, partition, load_attributes=False):
    """
    Parse DeepFashion items
    :param path_to_attributes: Path to the file with items' attributes
    :param path_to_bboxes: Path to the file with bounding boxes
    :param path_to_partition: Path to the file with partition information
    :param partition: Partition type
    :param load_attributes: Should load attributes (can take a while) or not
    :return: Dictionary with parsed item as value and path as key
    """
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


def get_attribute_names(item: Item, attrs: list):
    """
    :param item: Parsed item
    :param attrs: Attribute mapping to names
    :return: Names of attributes of given item
    """
    names = []
    for i, attr in enumerate(item.attributes):
        if attr == "1":
            names.append(attrs[i])
    return names


def parse_attribute_names(attr_types_path):
    """
    :param attr_types_path: Path to the file with attribute types
    :return: List of attribute names in corresponding order
    """
    names = []
    attribute_file = open(attr_types_path, "r", encoding="utf-8")
    lines_count = int(attribute_file.readline())
    attribute_file.readline()  # Header
    # Load attributes types
    for i in range(lines_count):
        attribute_data = re.split(r'\s\s+', attribute_file.readline(), 2)
        names.append(attribute_data[0])
    return names


def get_attr_type_mask(attr_types_path, type_ids):
    """
    Get indexes of attributes contained in given attribute types
    :param attr_types_path: Path to the file with attribute types
    :param type_ids: List of attribute type ids
    :return: List of attribute indexes
    """
    mask = []
    attribute_file = open(attr_types_path, "r", encoding="utf-8")
    lines_count = int(attribute_file.readline())
    attribute_file.readline()  # Header
    # Load attributes types
    for i in range(lines_count):
        attribute_data = re.split(r'\s\s+', attribute_file.readline(), 2)
        if attribute_data[1].rstrip() in map(lambda x: str(x), type_ids):
            mask.append(i)
    return mask


def get_attr_types(attr_types_path):
    """
    :param attr_types_path: Path to the file with attribute types
    :return: Dictionary with attribute index as key and attribute type id as value
    """
    types = {}
    attribute_file = open(attr_types_path, "r", encoding="utf-8")
    lines_count = int(attribute_file.readline())
    attribute_file.readline()  # Header
    # Load attributes types
    for i in range(lines_count):
        attribute_data = re.split(r'\s\s+', attribute_file.readline(), 2)
        types.update({i: (attribute_data[1].rstrip(), attribute_data[0])})
    return types


def prep_image(item: Item, data_dir: str) -> np.ndarray:
    """
    Prepare image for InceptionV3 model
    :param item: Parsed item to prepare
    :param data_dir: Dataset root directory
    :return: Array with preprocessed product image
    """
    img = image.load_img(data_dir + item.path)  # type: PILImage.Image
    img = img.crop(item.bbox)
    img = img.resize((299, 299))
    img_array = image.img_to_array(img)
    return tf.keras.applications.inception_v3.preprocess_input(img_array)


def get_embedding(model, items: dict, data_dir: str):
    """
    Creates items embedding
    :param model: Model used for feature extraction
    :param items: Dictionary fo parsed items
    :param data_dir: Dataset root directory
    :return: Tuple with embedding array and array of paths in corresponding order
    """
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
