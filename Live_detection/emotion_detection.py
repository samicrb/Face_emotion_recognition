import tensorflow as tf
import cv2
import numpy as np


# Charger le modèle Keras
model = tf.keras.models.load_model('FaceEmotionRecognition/models/emotion_model_final.keras')


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    return reshaped

def get_emotion_from_label(label):
    emotion_labels = ['Colère', 'Dégoût', 'Peur', 'Joie', 'Tristesse', 'Surprise', 'Neutre']
    return emotion_labels[np.argmax(label)]

# Charger le détecteur de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    """
    Détecte les visages dans une image.

    Arguments:
    frame -- Image d'entrée dans laquelle détecter les visages.

    Retourne:
    Une liste de rectangles englobants pour les visages détectés.
    Chaque rectangle est représenté par (x, y, w, h).
    """
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Détecter les visages dans l'image en niveaux de gris
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Retourner les rectangles des visages détectés
    return faces

# Ouvrir la vidéo
video_capture = cv2.VideoCapture('Live_detection/exemple_video.mp4')

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Détecter les visages
    faces = detect_faces(frame)
    
    for (x, y, w, h) in faces:
        # Extraire le visage
        face = frame[y:y+h, x:x+w]
        
        # Prétraiter l'image du visage
        preprocessed_face = preprocess_image(face)
        
        # Prédire l'émotion
        prediction = model.predict(preprocessed_face)
        emotion = get_emotion_from_label(prediction)
        
        # Dessiner le rectangle autour du visage
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Afficher l'émotion prédite
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Afficher la frame
    cv2.imshow('Video', frame)
    
    # Quitter la vidéo en appuyant sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
video_capture.release()
cv2.destroyAllWindows()
