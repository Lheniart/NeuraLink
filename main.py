import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Charger et préparer les données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normaliser les données (mettre les pixels entre 0 et 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Afficher les dimensions des données
print("Taille des données d'entraînement :", x_train.shape)
print("Taille des labels :", y_train.shape)

# Créer le modèle
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Couche d'entrée : transforme l'image 28x28 en vecteur 1D
    Dense(128, activation='relu'),  # Couche cachée : 128 neurones avec activation ReLU
    Dense(10, activation='softmax')  # Couche de sortie : 10 neurones pour les 10 classes (0 à 9)
])

# Compiler le modèle
model.compile(optimizer='adam',  # Optimiseur pour ajuster les poids
              loss='sparse_categorical_crossentropy',  # Fonction de perte pour classification multi-classes
              metrics=['accuracy'])  # Mesure de la précision

# Entraîner le modèle
print("Entraînement du modèle...")
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))  # Entraînement pendant 5 époques

# Évaluer le modèle sur les données de test
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nPrécision sur les données de test : {test_accuracy * 100:.2f}%")

# Prédire avec une image du jeu de test
image = x_test[0]  # Première image du jeu de test
prediction = model.predict(np.expand_dims(image, axis=0))  # Ajouter une dimension pour le batch
predicted_label = np.argmax(prediction)  # Trouver la classe avec la probabilité la plus élevée

# Afficher l'image et la prédiction
plt.imshow(image, cmap='gray')
plt.title(f"Prédiction : {predicted_label}")
plt.show()
