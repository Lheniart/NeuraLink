import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from keras.src.saving import load_model

# Chargement des modèles
model_1 = load_model("mnist_model.h5")
model_2 = load_model("mnist_cnn_model.h5")
model_3 = load_model("mnist_cnn_augmented.h5")
model_4 = load_model("mnist_cnn_augmented_reversed.h5")


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Niveaux de gris
    image = image.resize((28, 28))  # Redimension
    image_array = np.array(image) / 255.0  # Normalisation
    image_array = 1 - image_array  # Inversion
    image_array = np.expand_dims(image_array, axis=0)  # (1, 28, 28)
    return image_array


def predict(image_array):
    preds = []
    for model in (model_1, model_2, model_3, model_4):
        prediction = model.predict(image_array)
        preds.append(np.argmax(prediction))
    return preds


def choose_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Images PNG", "*.png"), ("Toutes les images", "*.jpg *.jpeg *.bmp *.gif")])
    if not file_path:
        return

    try:
        image_array = preprocess_image(file_path)
        predictions = predict(image_array)

        # Affichage de l'image dans la fenêtre
        raw_image = Image.open(file_path).resize((140, 140))
        img_tk = ImageTk.PhotoImage(raw_image)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Affichage des prédictions
        result_label.config(
            text=f"Prédiction modèle 1 : {predictions[0]}\n"
                 f"Prédiction modèle 2 : {predictions[1]}\n"
                 f"Prédiction modèle 3 : {predictions[2]}\n"
                 f"Prédiction modèle 4 : {predictions[3]}"
        )
    except Exception as e:
        messagebox.showerror("Erreur", str(e))


# Interface Tkinter
root = tk.Tk()
root.title("Reconnaissance de chiffres MNIST")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

btn = tk.Button(frame, text="Choisir une image", command=choose_file)
btn.pack()

image_label = tk.Label(frame)
image_label.pack(pady=10)

result_label = tk.Label(frame, text="", font=("Arial", 12))
result_label.pack()

root.mainloop()
