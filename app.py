# app.py
import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "final_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")
IMG_SIZE = (128, 128)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------
# Load Model & Labels
# -------------------------
print("Loading trained model...")
model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

idx_to_class = {v: k for k, v in labels.items()}

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)  # Enable CORS for all routes

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))

    result = {
        "summary": f"Predicted as {idx_to_class[predicted_class]}",
        "confidence": round(confidence, 2),
        "note": "This is an educational screening, not a clinical diagnosis."
    }

    return jsonify(result)

# NOTE: If you see "ModuleNotFoundError: No module named 'flask'" run:
#   ./setup.sh
# or:
#   python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)