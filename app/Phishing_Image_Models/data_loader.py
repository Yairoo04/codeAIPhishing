import os
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

def load_dataset(dataset_path, image_size=(128, 128), batch_size=32):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        label_mode='binary',  # Nhãn nhị phân: 0 (legit), 1 (phishing)
        image_size=image_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=42
    )
    
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        label_mode='binary',
        image_size=image_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=42
    )
    
    return train_dataset, val_dataset

def load_dataset_paths(dataset_path):
    image_paths = []
    labels = []

    for label, folder in enumerate(["legitimate", "phishing"]):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            print(f"Cảnh báo: Không tìm thấy thư mục {folder_path}, bỏ qua!")
            continue

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")): 
                image_paths.append(os.path.join(folder_path, filename))
                labels.append(label)

    return image_paths, labels

def preprocess_image(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Lỗi: Không thể đọc ảnh {image_path}")
    
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0 
    return img

