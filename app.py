from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Create Flask app
app = Flask(__name__, template_folder='.')  # '.' means current folder

# Load trained model
model = load_model("mask_model.h5")

# Route to serve HTML page
@app.route('/')
def home_page():
    return render_template('index.html')

# API route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    img = Image.open(file).convert('RGB')
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)
    label = "Mask" if np.argmax(pred) == 1 else "No Mask"
    confidence = float(np.max(pred))

    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
