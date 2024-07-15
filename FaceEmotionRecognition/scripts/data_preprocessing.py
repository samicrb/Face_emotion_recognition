import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Répertoires
raw_data_dir = 'FaceEmotionRecognition/data/raw'
processed_data_dir = 'FaceEmotionRecognition/data/processed'

# Chargement des données
def load_data(file_path):
    data = pd.read_csv(file_path)
    pixels = data['pixels'].tolist()
    images = np.array([np.fromstring(pixel, dtype=int, sep=' ').reshape(48, 48) for pixel in pixels])
    labels = pd.get_dummies(data['emotion']).values
    return images, labels

# Redimensionnement des images
def resize_images(images, size=(48, 48)):
    resized_images = np.array([cv2.resize(image, size) for image in images])
    return resized_images

# Normalisation des pixels
def normalize_images(images):
    return images / 255.0

# Ajout d'une dimension de canal
def add_channel_dimension(images):
    return np.expand_dims(images, axis=-1)

# Augmentation des données
def augment_data(images, labels):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(images)
    return datagen

# Enregistrement des données prétraitées
def save_data(images, labels, prefix):
    np.save(os.path.join(processed_data_dir, f'{prefix}_images.npy'), images)
    np.save(os.path.join(processed_data_dir, f'{prefix}_labels.npy'), labels)

# Prétraitement principal
def preprocess_data():
    images, labels = load_data(os.path.join(raw_data_dir, 'fer2013.csv'))
    images = resize_images(images)
    images = normalize_images(images)
    images = add_channel_dimension(images)  # Ajout de la dimension de canal
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    datagen = augment_data(X_train, y_train)
    save_data(X_train, y_train, 'train')
    save_data(X_test, y_test, 'test')
    print('Données prétraitées et enregistrées.')

if __name__ == "__main__":
    preprocess_data()
