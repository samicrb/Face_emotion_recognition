from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Charger le meilleur modèle sauvegardé
model = load_model('models/emotion_model.keras')


x_test = np.load('data/processed/test_images.npy')
y_test = np.load('data/processed/test_labels.npy')
# Évaluer le modèle sur les données de test
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Loss sur le test: {test_loss}')
print(f'Précision sur le test: {test_accuracy}')

# Prédictions sur les données de test
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Générer la matrice de confusion
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Afficher la matrice de confusion
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prédictions')
plt.ylabel('Véritables')
plt.title('Matrice de confusion')
plt.show()

# Générer le rapport de classification
class_report = classification_report(y_true, y_pred_classes, target_names=['Colère', 'Dégoût', 'Peur', 'Bonheur', 'Tristesse', 'Surprise', 'Neutre'])
print(class_report)
