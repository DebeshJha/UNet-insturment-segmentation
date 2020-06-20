
import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from utils import create_dir, shuffle
from data import load_data

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    ChannelShuffle,
    CoarseDropout
)

def augment_data(files, save_path, augment=True):
    """ Performing data augmentation. """
    # original_size = (960, 540) 640x360
    size = (640, 360)
    crop_size = (640, 360)

    files_name = ["instrument_instances.png", "raw.png"]
    for idx, path in tqdm(enumerate(files), total=len(files)):
        f = os.listdir(path)
        f.sort()
        len_f = len(f)

        ## Reading Image and Mask
        image = os.path.join(path, files_name[1])
        mask = os.path.join(path, files_name[0])

        image_name = "_".join([d for d in image.split("..")[-1].split("/")])
        image_name = image_name.split("_ml_dataset_")[-1].split(".")[0]

        mask_name = "_".join([d for d in mask.split("..")[-1].split("/")])
        mask_name = mask_name.split("_ml_dataset_")[-1].split(".")[0]

        x = cv2.imread(image, cv2.IMREAD_COLOR)
        if len_f == 1:
            y = np.zeros((540, 960, 3))
        else:
            y = cv2.imread(mask, cv2.IMREAD_COLOR) * 255.0

        h, w, c = x.shape

        if augment == True:

            ## Crop
            x_min = 0
            y_min = 0
            x_max = x_min + size[0]
            y_max = y_min + size[1]

            aug = Crop(p=1, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
            augmented = aug(image=x, mask=y)
            x1 = augmented['image']
            y1 = augmented['mask']

            ## Random Rotate 90 degree
            aug = RandomRotate90(p=1)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            ## ElasticTransform
            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            ## Grid Distortion
            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            ## Optical Distortion
            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            ## Vertical Flip
            aug = VerticalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x6 = augmented['image']
            y6 = augmented['mask']

            ## Horizontal Flip
            aug = HorizontalFlip(p=1)
            augmented = aug(image=x, mask=y)
            x7 = augmented['image']
            y7 = augmented['mask']

            ## Grayscale
            x8 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            y8 = y

            ## Grayscale Vertical Flip
            aug = VerticalFlip(p=1)
            augmented = aug(image=x8, mask=y8)
            x9 = augmented['image']
            y9 = augmented['mask']

            ## Grayscale Horizontal Flip
            aug = HorizontalFlip(p=1)
            augmented = aug(image=x8, mask=y8)
            x10 = augmented['image']
            y10 = augmented['mask']

            ##
            # aug = RandomBrightnessContrast(p=1)
            # augmented = aug(image=x, mask=y)
            # x11 = augmented['image']
            # y11 = augmented['mask']
            #
            # aug = RandomGamma(p=1)
            # augmented = aug(image=x, mask=y)
            # x12 = augmented['image']
            # y12 = augmented['mask']
            #
            # aug = HueSaturationValue(p=1)
            # augmented = aug(image=x, mask=y)
            # x13 = augmented['image']
            # y13 = augmented['mask']
            #
            # aug = RGBShift(p=1)
            # augmented = aug(image=x, mask=y)
            # x14 = augmented['image']
            # y14 = augmented['mask']
            #
            # aug = RandomBrightness(p=1)
            # augmented = aug(image=x, mask=y)
            # x15 = augmented['image']
            # y15 = augmented['mask']
            #
            # aug = RandomContrast(p=1)
            # augmented = aug(image=x, mask=y)
            # x16 = augmented['image']
            # y16 = augmented['mask']
            #
            # aug = ChannelShuffle(p=1)
            # augmented = aug(image=x, mask=y)
            # x17 = augmented['image']
            # y17 = augmented['mask']

            aug = CoarseDropout(p=1, max_holes=8, max_height=32, max_width=32)
            augmented = aug(image=x, mask=y)
            x18 = augmented['image']
            y18 = augmented['mask']

            # aug = GaussNoise(p=1)
            # augmented = aug(image=x, mask=y)
            # x19 = augmented['image']
            # y19 = augmented['mask']
            #
            # aug = MotionBlur(p=1, blur_limit=7)
            # augmented = aug(image=x, mask=y)
            # x20 = augmented['image']
            # y20 = augmented['mask']
            #
            # aug = MedianBlur(p=1, blur_limit=10)
            # augmented = aug(image=x, mask=y)
            # x21 = augmented['image']
            # y21 = augmented['mask']
            #
            # aug = GaussianBlur(p=1, blur_limit=10)
            # augmented = aug(image=x, mask=y)
            # x22 = augmented['image']
            # y22 = augmented['mask']

            images = [
                x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10,
                #x11, x12, x14, x15, x16, x17,
                x18
                # x19, x20, x21, x22
            ]
            masks  = [
                y, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10,
                # y10, y11, y12, y14, y15, y16, y17,
                y18
                # y19, y20, y21, y22
            ]

        else:
            images = [x]
            masks  = [y]

        idx = 0
        for i, m in zip(images, masks):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{image_name}_{idx}.jpg"
            tmp_mask_name  = f"{mask_name}_{idx}.jpg"

            image_path = os.path.join(save_path, "image/", tmp_image_name)
            mask_path  = os.path.join(save_path, "mask/", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1

def main():
    np.random.seed(42)
    path = "../../../../ml_dataset/Instrument/"
    train_path, valid_path, test_path = load_data(path)

    print("Train: ", len(train_path))
    print("Valid: ", len(valid_path))
    print("Test: ", len(test_path))

    create_dir("../new_data/train/image/")
    create_dir("../new_data/train/mask/")
    create_dir("../new_data/valid/image/")
    create_dir("../new_data/valid/mask/")
    create_dir("../new_data/test/image/")
    create_dir("../new_data/test/mask/")

    augment_data(train_path, "../new_data/train/", augment=False)
    augment_data(valid_path, "../new_data/valid/", augment=False)
    augment_data(test_path, "../new_data/test/", augment=False)

if __name__ == "__main__":
    main()
