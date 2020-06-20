
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision, MeanIoU
from glob import glob
from sklearn.model_selection import train_test_split
from utils import read_params, shuffling, create_dir
from metrics import dice_loss, dice_coef, iou
from data import tf_dataset
from sgdr import SGDRScheduler
from unet import UNet

if __name__ == "__main__":
    print("")
    ## Seeding
    np.random.seed(42)
    tf.random.set_seed(42)

    ## Remove
    os.system("rm files/files.csv")
    os.system("rm logs")

    ## Creating folders
    create_dir("files")
    create_dir("results")

    ## Dataset
    train_path = "../new_data/train/"
    valid_path = "../new_data/valid/"

    ## Training
    train_x = sorted(glob(os.path.join(train_path, "image", "*.jpg")))
    train_y = sorted(glob(os.path.join(train_path, "mask", "*.jpg")))

    ## Shuffling
    train_x, train_y = shuffling(train_x, train_y)

    ## Validation
    valid_x = sorted(glob(os.path.join(valid_path, "image", "*.jpg")))
    valid_y = sorted(glob(os.path.join(valid_path, "mask", "*.jpg")))

    H = 360
    W = 640
    batch_size = 2
    lr = 1e-3
    epochs = 200
    model_path = "files/model.h5"
    csv_path = "files/data.csv"

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    model = UNet(H, W)
    metrics = [dice_coef, iou, MeanIoU(num_classes=2), Recall(), Precision()]
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)
    model.summary()

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
        SGDRScheduler(min_lr=1e-6, max_lr=1e-3, steps_per_epoch=np.ceil(epochs/batch_size),
         lr_decay=0.9, cycle_length=5, mult_factor=1.5)
    ]

    train_steps = (len(train_x)//batch_size)
    valid_steps = (len(valid_x)//batch_size)

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    model.fit(train_dataset,
            epochs=epochs,
            validation_data=valid_dataset,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            callbacks=callbacks,
            shuffle=False)
