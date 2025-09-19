from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import json
import os

app = Flask(__name__)
CORS(app)

# Load model
MODEL_FILE = "breed_recognition_model.h5"
CLASS_FILE = "breed_classes.json"
model = tf.keras.models.load_model(MODEL_FILE)

with open(CLASS_FILE, "r") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

def preprocess_image(image_data):
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes)).resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        image_data_base64 = data['image_data'].split(",")[1]
        img = preprocess_image(image_data_base64)

        # 🔹 Prediction

        preds = model.predict(img)
        class_index = np.argmax(preds)
        confidence = float(preds[0][class_index])

        # 🔹 Apply threshold
        if confidence < 0.8:  
        # Force confidence to a low value
         confidence = round(np.random.uniform(0.15, 0.25), 2)  
         breed_name = "Breed Not Found"
        else:
         breed_name = class_names[class_index]

        return jsonify({"breedName": breed_name, "confidenceScore": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

