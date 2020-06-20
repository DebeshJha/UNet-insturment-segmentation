
import os
import json
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.metrics import Recall, Precision, MeanIoU
from tensorflow.keras.optimizers import Adam
from metrics import iou, dice_coef, dice_loss

def read_params():
    """ Reading the parameters from the JSON file."""
    with open("params.json", "r") as f:
        data = f.read()
        params = json.loads(data)
        return params

def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_model(path, compile=True, lr=1e-3):
    with CustomObjectScope({
            'iou':iou,
            'dice_coef':dice_coef,
            'dice_loss':dice_loss
        }):
        model = tf.keras.models.load_model(path)

        if compile:
            metrics = [dice_coef, iou, MeanIoU(num_classes=2), Recall(), Precision()]
            model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)
        return model
