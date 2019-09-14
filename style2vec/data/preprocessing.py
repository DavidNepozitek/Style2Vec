from tensorflow.python.keras.preprocessing.image import array_to_img, img_to_array
from PIL import Image
import numpy as np


def resize(item: np.ndarray, height: int, width: int, keep_ratio: bool = False) -> np.ndarray:
    """

    Parameters
    ----------
    item
        Input array
    height
        Target height
    width
        Target width
    keep_ratio
        Keep image aspect ratio

    Returns
    -------
    np.ndarray
        Resized image in numpy array
    """
    img = array_to_img(item, scale=False)
    if keep_ratio:
        img.thumbnail((width, width), Image.ANTIALIAS)
        resized_img = img
    else:
        resized_img = img.resize((width, height), resample=Image.NEAREST)

    resized_img = img_to_array(resized_img)
    resized_img = resized_img.astype(dtype=np.uint8)

    return resized_img



