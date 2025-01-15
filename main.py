import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Chemin principal des données
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
print(f"Noms des classes : {class_names}")

# 2. Normaliser les données
normalization_layer = layers.Rescaling(1./255)  # Normalisation des pixels
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Améliorer les performances avec mise en cache et préchargement
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# 3. Construire un modèle CNN
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),  # Définir explicitement l'entrée
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),  # Ajouter une couche dense avec plus de neurones
    layers.Dropout(0.5),  # Ajouter Dropout pour réduire l'overfitting
    layers.Dense(len(class_names), activation='softmax')  # Nombre de classes
])

# 4. Compiler le modèle
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Entraîner le modèle
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=25 # Ajustez selon vos besoins
)

# 6. Sauvegarder le modèle
model.save("dog_breed_classifier.h5")
print("Modèle sauvegardé avec succès.")

# 7. Tester le modèle avec une image
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
