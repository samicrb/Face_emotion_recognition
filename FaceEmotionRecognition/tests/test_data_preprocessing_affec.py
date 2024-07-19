from test_data_preprocessing import display_samples
import os
import numpy as np

# Répertoires
processed_data_dir = 'FaceEmotionRecognition/data/second_training/processed'

# Chargement des données
X_train = np.load(os.path.join(processed_data_dir, 'affect_train_images.npy'))
y_train = np.load(os.path.join(processed_data_dir, 'affect_train_labels.npy'))

display_samples(X_train, y_train)