
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tqdm import tqdm
from glob import glob
from data import load_data, tf_dataset
from utils import create_dir, load_model

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    # x = cv2.resize(x, (256, 256))
    x = x/255.0
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # x = cv2.resize(x, (256, 256))
    x = np.expand_dims(x, axis=-1)
    return x

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

def evaluate_normal(model, x_data, y_data):
    THRESHOLD = 0.5
    total = []
    for i, (x, y) in tqdm(enumerate(zip(x_data, y_data)), total=len(x_data)):
        name = x_data[i].split("/")[-1]

        x = read_image(x)
        y = read_mask(y)

        y_pred = model.predict(np.expand_dims(x, axis=0))[0]

        h, w, _ = x.shape
        line = np.ones((h, 10, 3)) * 255.0

        all_images = [
            x * 255.0, line,
            mask_parse(y), line,
            mask_parse(y_pred) * 255.0
        ]
        mask = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"results/{name}", mask)

if __name__ == "__main__":
    print("")
    ## Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    ## Creating folders
    create_dir("results/")

    ## Hyperparameters
    batch_size = 32

    test_path = "../new_data/test/"
    test_x = sorted(glob(os.path.join(test_path, "image", "*.jpg")))
    test_y = sorted(glob(os.path.join(test_path, "mask", "*.jpg")))
    test_dataset = tf_dataset(test_x, test_y, batch=batch_size)

    test_steps = (len(test_x)//batch_size)
    if len(test_x) % batch_size != 0:
        test_steps += 1

    model = load_model("files/model.h5")

    model.evaluate(test_dataset, steps=test_steps)
    evaluate_normal(model, test_x, test_y)
