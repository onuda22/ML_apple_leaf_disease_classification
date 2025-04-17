import tkinter as tk
import numpy as np
import tensorflow as tf
import json

from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk

# Load Model
model = tf.keras.models.load_model("models/model.h5")

# Input size
IMG_SIZE=(224,224)

# Class Labels
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

classes = [label for label, _ in sorted(class_indices.items(), key=lambda x: x[1])]

# Process Image
def image_processing(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict 
def predict_image(image_path):
    img_array = image_processing(image_path)
    prediction = model.predict(img_array)
    return prediction

# Upload Image
def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    # Show Image
    img = Image.open(file_path)
    img.thumbnail((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    panel.configure(image=img_tk)
    panel.image = img_tk

    # make prediction
    prediction = predict_image(file_path)

    # format output
    pred_classes = np.argmax(prediction)
    confidence = np.max(prediction)
    class_label = classes[pred_classes]
    result = f"Predicted as: {class_label} (confidence: {confidence:.4f})"

    label_result.config(text=result)


# --- GUI SETUP ---
root = tk.Tk()
root.title("Apple Leaf Disease Prediction")

btn_upload = tk.Button(root, text="Upload Image", command=upload_image)
btn_upload.pack(pady=10)

panel = Label(root)
panel.pack()

label_result = Label(root, text="", font=("Helvetica", 14))
label_result.pack(pady=10)

root.mainloop()