from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist


# Charger et préparer les données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape pour être compatible avec les CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Normalisation entre 0 et 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Création du modèle CNN avec Data Augmentation
model = Sequential([
    # Première couche convolutionnelle
    Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    # Deuxième couche convolutionnelle
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    # Troisième couche convolutionnelle
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    Dropout(0.25),

    # Quatrième couche convolutionnelle
    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    Dropout(0.25),

    # Passage en 1D
    Flatten(),

    # Première couche dense
    Dense(512, activation='relu'),
    Dropout(0.5),  # Réduit l’overfitting

    # Deuxième couche dense
    Dense(256, activation='relu'),
    Dropout(0.5),

    # Couche de sortie
    Dense(10, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#data Augmentation
datagen = ImageDataGenerator(
    rotation_range=180,  # Rotation aléatoire jusqu'à 180°
    width_shift_range=0.1,  # Décalage horizontal (10%)
    height_shift_range=0.1,  # Décalage vertical (10%)
    zoom_range=0.1  # Zoom aléatoire (10%)
)

# Entraînement du modèle avec Data Augmentation
print("Entraînement")
model.fit(datagen.flow(x_train, y_train, batch_size=32),
          epochs=15,
          validation_data=(x_test, y_test))

# Sauvegarde du modèle
model.save("mnist_cnn_augmented.h5")
print("Modèle sauvegardé sous 'mnist_cnn_augmented.h5'.")


