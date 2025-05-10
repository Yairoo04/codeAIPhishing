import os
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42


def load_dataset(dataset_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                 val_split=VAL_SPLIT, seed=SEED):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels='inferred',
        label_mode='binary',
        validation_split=val_split,
        subset='training',
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels='inferred',
        label_mode='binary',
        validation_split=val_split,
        subset='validation',
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )
    return train_ds, val_ds
