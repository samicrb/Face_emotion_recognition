import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Charger le modèle
model = load_model('models/emotion_model.keras')

# Dictionnaire des étiquettes
labels_dict = {0: 'Colère', 1: 'Dégoût', 2: 'Peur', 3: 'Bonheur', 4: 'Tristesse', 5: 'Surprise', 6: 'Neutre'}

st.title('Reconnaissance des Émotions')

uploaded_file = st.file_uploader("Choisissez une image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)
    st.write("")
    st.write("Classification...")

    # Prétraitement de l'image
    img_array = np.array(image.convert('L').resize((48, 48)))
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)

    # Prédiction
    predictions = model.predict(img_array)
    max_index = np.argmax(predictions[0])
    predicted_label = labels_dict[max_index]

    st.write(f"L'émotion prédite est : {predicted_label}")
