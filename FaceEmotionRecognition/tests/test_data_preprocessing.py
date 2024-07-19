import os
import numpy as np
import matplotlib.pyplot as plt

# Répertoires
processed_data_dir = 'FaceEmotionRecognition/data/processed'

# Chargement des données
X_train = np.load(os.path.join(processed_data_dir, 'train_images.npy'))
y_train = np.load(os.path.join(processed_data_dir, 'train_labels.npy'))

# Affichage de quelques exemples d'images et d'étiquettes
def display_samples(X, y, num_samples=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X[i], cmap='gray')
        plt.title(f"Label: {np.argmax(y[i])}")
        plt.axis('off')
    plt.show()

display_samples(X_train, y_train)
