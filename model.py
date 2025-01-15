import os

import tensorflow as tf


model = tf.keras.models.load_model("dog_breed_classifier.h5")

base_dir = r"C:\Users\LoanH\Downloads\archive\images\Images"  # Remplacez par le chemin de votre dataset

# 1. Charger les données depuis les dossiers avec image_dataset_from_directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,  # 20% des données pour la validation
    subset="training",
    seed=123,  # Pour reproduire la séparation train/val
    image_size=(150, 150),  # Taille des images
    batch_size=32  # Taille des lots
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(150, 150),
    batch_size=32
)

# Récupérer les noms des classes
class_names = train_dataset.class_names


print("Modèle chargé avec succès.")



# Fonction pour prédire la race d'une image
def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(150, 150))  # Charger et redimensionner l'image
    img_array = tf.keras.utils.img_to_array(img) / 255.0  # Normaliser
    img_array = tf.expand_dims(img_array, 0)  # Ajouter une dimension pour le batch

    predictions = model.predict(img_array)
    predicted_class = class_names[tf.argmax(predictions[0])]
    return predicted_class

# Exemple de prédiction
test_image_path = r"C:\Users\LoanH\Downloads\Images\n02085620-Chihuahua\n02085620_275.jpg"  # Remplacez par une image réelle
if os.path.exists(test_image_path):
    predicted_breed = predict_image(test_image_path)
    print(f"Race prédite : {predicted_breed}")
else:
    print(f"Image test introuvable : {test_image_path}")

