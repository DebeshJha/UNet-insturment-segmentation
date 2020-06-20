
import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import CustomObjectScope
from metrics import iou

# def load_data(path):
#     path1 = os.path.join(path, "Prokto/")
#     path2 = os.path.join(path, "Rektum/")
#     path1_dir = sorted(os.listdir(path1))
#
#     ## Train path
#     train_path = []
#     for p in path1_dir[0:6]:
#         tmp_path1 = sorted(glob(os.path.join(path, "Prokto/", p, "*/")))
#         tmp_path2 = sorted(glob(os.path.join(path, "Rektum/", p, "*/")))
#         train_path += tmp_path1 + tmp_path2
#
#     ## Valid path
#     tmp_path1 = sorted(glob(os.path.join(path, "Prokto/", path1_dir[6], "*/")))
#     tmp_path2 = sorted(glob(os.path.join(path, "Rektum/", path1_dir[6], "*/")))
#     valid_path = tmp_path1 + tmp_path2
#
#     ## Test path
#     tmp_path1 = sorted(glob(os.path.join(path, "Prokto/", path1_dir[7], "*/")))
#     tmp_path2 = sorted(glob(os.path.join(path, "Rektum/", path1_dir[7], "*/")))
#     test_path = tmp_path1 + tmp_path2
#
#     return train_path, valid_path, test_path

def load_data(path):
    path1 = sorted(glob(os.path.join(path, "Prokto/*/*")))
    path2 = sorted(glob(os.path.join(path, "Rektum/*/*")))
    paths = path1 + path2

    total = len(paths)
    split = int(0.1 * total)

    train_path, valid_path = train_test_split(paths, random_state=42, test_size=split)
    train_path, test_path = train_test_split(train_path, random_state=42, test_size=split)

    return train_path, valid_path, test_path

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    # x = cv2.resize(x, (512, 256))
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # x = cv2.resize(x, (512, 256))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    x = x.astype(np.float32)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([360, 640, 3])
    y.set_shape([360, 640, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset
