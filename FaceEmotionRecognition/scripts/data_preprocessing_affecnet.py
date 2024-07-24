import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Chemins vers les données
data_dir = 'FaceEmotionRecognition/data/second_training/raw/affectnet'  
csv_file = 'FaceEmotionRecognition/data/second_training/raw/affectnet/labels.csv'  

# Charger les labels
data = pd.read_csv(csv_file)

# Afficher les labels uniques avant la filtration
print("Labels before filtering:", data['label'].unique())

# Correspondance des labels entre Affectnet et votre modèle
label_map = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,  # Correction ici
    'sad': 4,    # Correction ici
    'surprise': 5,
    'neutral': 6
}

# Supprimer les labels 'contempt'
data = data[data['label'] != 'contempt']

# Filtrer les labels non utilisés
data = data[data['label'].isin(label_map.keys())]

# Vérification après la filtration
print("Labels after filtering:", data['label'].unique())
print("Label distribution:")
print(data['label'].value_counts())

# Fonction pour redimensionner les images
def resize_image(image, size=(48, 48)):
    return cv2.resize(image, size)

# Charger et prétraiter les images
images = []
labels = []

for index, row in data.iterrows():
    img_path = os.path.join(data_dir, row['pth'])
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Charger en niveaux de gris
    if image is not None:
        image = resize_image(image)
        image = image / 255.0  # Normalisation
        images.append(image)
        labels.append(label_map[row['label']])

# Conversion en tableaux numpy
images = np.array(images)
labels = pd.get_dummies(labels).values  # One-hot encoding des labels

# Ajouter une dimension de canal pour les images
images = np.expand_dims(images, axis=-1)

# Division des données en ensembles d'entraînement et de test
affec_x_train, affec_x_test, affec_y_train, affec_y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Sélectionner plus d'échantillons pour les classes moins performantes (0, 1, 2)
def augment_minority_classes(X, y, class_indices, factor=2):
    augmented_X = X.copy()
    augmented_y = y.copy()
    for class_index in class_indices:
        class_samples_X = X[np.argmax(y, axis=1) == class_index]
        class_samples_y = y[np.argmax(y, axis=1) == class_index]
        for _ in range(factor - 1):
            augmented_X = np.concatenate((augmented_X, class_samples_X), axis=0)
            augmented_y = np.concatenate((augmented_y, class_samples_y), axis=0)
    return augmented_X, augmented_y

affec_x_train, affec_y_train = augment_minority_classes(affec_x_train, affec_y_train, [0, 1, 2], factor=2)

# Mélanger les données pour éviter des patterns lors de l'entraînement
indices = np.arange(affec_x_train.shape[0])
np.random.shuffle(indices)
affec_x_train = affec_x_train[indices]
affec_y_train = affec_y_train[indices]

# Enregistrement des données prétraitées
np.save('FaceEmotionRecognition/data/second_training/processed/affect_train_images.npy', affec_x_train)
np.save('FaceEmotionRecognition/data/second_training/processed/affect_train_labels.npy', affec_y_train)
np.save('FaceEmotionRecognition/data/second_training/processed/affect_test_images.npy', affec_x_test)
np.save('FaceEmotionRecognition/data/second_training/processed/affect_test_labels.npy', affec_y_test)

print("Prétraitement des données terminé et sauvegardé.")
