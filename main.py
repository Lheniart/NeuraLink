# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import History

# Charger les données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisation des images (passer de [0, 255] à [0, 1])
x_train, x_test = x_train / 255.0, x_test / 255.0

# Redimensionner les images pour qu'elles aient une seule dimension pour les canaux (28, 28, 1)
x_train = x_train[..., None]
x_test = x_test[..., None]

# Encoder les labels (de [0, 1, ..., 9] à [10 classes en one-hot encoding])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Affichage des premières images d'entraînement pour vérification
def plot_images(images, labels, predictions=None):
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {np.argmax(labels[i])}')
        if predictions is not None:
            plt.xlabel(f'Pred: {np.argmax(predictions[i])}')
        plt.axis('off')
    plt.show()

# Visualiser quelques images d'entraînement
plt.title('Affichage des premières images d\'entraînement pour vérification')
plot_images(x_train, y_train)

# Construction du modèle CNN
model = Sequential([
    Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes pour les chiffres de 0 à 9
])

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle avec suivi de l'historique
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=64)

# Évaluer le modèle sur le jeu de test
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Précision sur le jeu de test : {accuracy:.2f}")

# Affichage des courbes de perte et de précision
def plot_training_history(history):
    # Courbes de perte
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perte d\'entraînement')
    plt.plot(history.history['val_loss'], label='Perte de validation')
    plt.title('Courbe de perte')
    plt.xlabel('Epochs')
    plt.ylabel('Perte')
    plt.legend()

    # Courbes de précision
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Précision d\'entraînement')
    plt.plot(history.history['val_accuracy'], label='Précision de validation')
    plt.title('Courbe de précision')
    plt.xlabel('Epochs')
    plt.ylabel('Précision')
    plt.legend()

    plt.show()

# Visualiser les courbes d'entraînement
plt.title('Affichage des courbes de perte et de précision')
plot_training_history(history)

# Prédictions sur les données de test
predictions = model.predict(x_test)

# Visualiser les premières prédictions
plt.title('Affichage des premières prédictions du modèle')
plot_images(x_test, y_test, predictions)