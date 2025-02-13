from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist

# Charger les données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape des images pour les CNN (ajout de la dimension des canaux : 1)
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0  # Normalisation
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0  # Normalisation

# Créer un modèle CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),  # 32 filtres, kernel 3x3
    MaxPooling2D((2,2)),  # Réduction de dimension
    Conv2D(64, (3,3), activation='relu'),  # Deuxième couche de convolution
    MaxPooling2D((2,2)),
    Flatten(),  # Passage en 1D
    Dense(128, activation='relu'),
    Dropout(0.5),  # Évite l'overfitting
    Dense(10, activation='softmax')  # 10 classes (0-9)
])

# Compilation du modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Sauvegarde du modèle
#model.save("mnist_cnn_model.h5")
