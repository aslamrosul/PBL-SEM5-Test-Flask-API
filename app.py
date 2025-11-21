import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from skimage.feature import hog


app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("model_svm.pkl")
classes = ["menjari", "menyirip", "melengkung", "sejajar"]


def extract_color_features(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    R_mean = np.mean(img_rgb[:, :, 0])
    G_mean = np.mean(img_rgb[:, :, 1])
    B_mean = np.mean(img_rgb[:, :, 2])
    return np.array([R_mean, G_mean, B_mean])


def extract_features(img):
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feature = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    color_feat = extract_color_features(img)

    combined = np.concatenate([hog_feature, color_feat])
    return combined.reshape(1, -1)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image not found"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    features = extract_features(img)

    pred_idx = model.predict(features)[0]
    pred_class = classes[pred_idx]

    return jsonify({"prediction": pred_class})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
