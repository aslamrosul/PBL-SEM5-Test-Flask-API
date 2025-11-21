#model hayyin
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from skimage.feature import hog
from skimage.color import rgb2gray

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("svm_daun_model.pkl")
classes = ["melengkung", "menjari", "menyirip", "sejajar"]


# =========================
#   HOMOMORPHIC FILTER
# =========================
def homomorphic_filter(img_gray):
    img_log = np.log1p(img_gray)
    M, N = img_log.shape
    sigma = 30

    X, Y = np.mgrid[0:M, 0:N]
    center_x, center_y = M/2, N/2
    gaussian = np.exp(-((X-center_x)**2 + (Y-center_y)**2) / (2*sigma*sigma))
    H = 1 - gaussian

    img_fft = np.fft.fft2(img_log)
    img_filt = np.fft.ifft2(img_fft * H)
    img_exp = np.exp(np.real(img_filt))
    img_norm = cv2.normalize(img_exp, None, 0, 1, cv2.NORM_MINMAX)

    return img_norm


# =========================
#   HSV COLOR HIST
# =========================
def hsv_features(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist([hsv_image], [0, 1], None,
                              [16, 8], [0, 180, 0, 256])
    cv2.normalize(color_hist, color_hist, 0, 1, cv2.NORM_MINMAX)
    return color_hist.flatten()


# =========================
#   FULL FEATURE EXTRACTOR
# =========================
def extract_features(img):
    img = cv2.resize(img, (256, 256))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = rgb2gray(img_rgb)

    # Gamma correction
    gamma = 0.7
    gray_gamma = np.power(gray, gamma)

    # Homomorphic filter
    gray_homo = homomorphic_filter(gray_gamma)

    # Gaussian blur
    blur = cv2.GaussianBlur((gray_homo * 255).astype(np.uint8), (3, 3), 0)

    # Canny edge
    edges = cv2.Canny(blur, 40, 120)

    # HOG
    hog_features = hog(
        edges,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    # HSV
    hsv_feat = hsv_features(img)

    # Hybrid (HOG + HSV)
    hybrid = np.concatenate([hog_features, hsv_feat])

    return hybrid.reshape(1, -1)


# =========================
#      ROUTE PREDICT
# =========================
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
