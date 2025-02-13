import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.src.saving import load_model

# Charger une image personnelle
image_path = "C:/Users/LoanH/Downloads/chiffre_2.png"  # Remplace par ton chemin d'image

# Ouvrir l'image avec PIL
image = Image.open(image_path).convert("L")  # Convertir en niveaux de gris

# Redimensionner en 28x28 pixels
image = image.resize((28, 28))

# Convertir en numpy array
image_array = np.array(image)

# Normaliser les pixels (0-255 → 0-1)
image_array = image_array / 255.0

# Inverser les couleurs (si fond blanc et chiffre noir)
image_array = 1 - image_array

# Afficher l'image après prétraitement
plt.imshow(image_array, cmap="gray")
plt.title("Image après prétraitement")
plt.show()

model_1 = load_model("mnist_model.h5")
model_2 = load_model("mnist_cnn_model.h5")
model_3 = load_model("mnist_cnn_augmented.h5")

image_array = np.expand_dims(image_array, axis=0)  # (1, 28, 28)

prediction = model_1.predict(image_array)
predicted_label = np.argmax(prediction)

print(f"Prédiction du modèle 1 : {predicted_label}")

prediction = model_2.predict(image_array)
predicted_label = np.argmax(prediction)

print(f"Prédiction du modèle 2 : {predicted_label}")

prediction = model_3.predict(image_array)
predicted_label = np.argmax(prediction)

print(f"Prédiction du modèle 3 : {predicted_label}")


