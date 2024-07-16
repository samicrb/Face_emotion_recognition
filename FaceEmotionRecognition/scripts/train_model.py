import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

# Charger les données
x_train = np.load('data/processed/train_images.npy')
y_train = np.load('data/processed/train_labels.npy')
x_test = np.load('data/processed/test_images.npy')
y_test = np.load('data/processed/test_labels.npy')

print("Shape des données d'entraînement :", x_train.shape)
print("Shape des labels d'entraînement :", y_train.shape)
print("Shape des données de validation :", x_test.shape)
print("Shape des labels de validation :", y_test.shape)


def create_cnn_model(input_shape):
    model = Sequential()
    # Application des filtres de convolution 
    # Extraction des contours et les textures
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # Diminution du nombre de paramètres et la charge computationnelle.
    model.add(MaxPooling2D((2, 2)))
    #Prévention sur-apprentissage
    model.add(Dropout(0.25))

    #Deuxième couche convolutive
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Troisième couche convolutive
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Couche de flattening 2D to 1D
    model.add(Flatten())
    
    # Couche dense
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # Couche de sortie
    model.add(Dense(7, activation='softmax'))  # 7 classes d'émotions

    return model

model = create_cnn_model((48, 48, 1))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Définition des callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('models/emotion_model.keras', save_best_only=True)

# Entraînement du modèle
history = model.fit(x_train, y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping, model_checkpoint])

# Sauvegarder le modèle final
model.save('models/emotion_model_final.keras')

# Évaluer le modèle sur les données de validation
val_loss, val_accuracy = model.evaluate(x_test, y_test)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')
