import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from skimage.feature import hog
from skimage.color import rgb2gray

app = Flask(__name__)
CORS(app)

# ============================
# LOAD DUA MODEL
# ============================
model_hayyin = joblib.load("svm_daun_model.pkl")
model_naufal = joblib.load("model_svm.pkl")

classes_hayyin = ["melengkung", "menjari", "menyirip", "sejajar"]
classes_naufal = ["menjari", "menyirip", "melengkung", "sejajar"]


# =====================================================
#   PREPROCESSING & FEATURE EXTRACTION — MODEL HAYYIN
# =====================================================

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


def hsv_features(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_hist = cv2.calcHist(
        [hsv_image],
        [0, 1],
        None,
        [16, 8],
        [0, 180, 0, 256]
    )
    cv2.normalize(color_hist, color_hist, 0, 1, cv2.NORM_MINMAX)
    return color_hist.flatten()


def extract_features_hayyin(img):
    img = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = rgb2gray(img_rgb)

    gamma = 0.7
    gray_gamma = np.power(gray, gamma)

    gray_homo = homomorphic_filter(gray_gamma)

    blur = cv2.GaussianBlur((gray_homo * 255).astype(np.uint8), (3, 3), 0)

    edges = cv2.Canny(blur, 40, 120)

    hog_features = hog(
        edges,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    hsv_feat = hsv_features(img)

    final_features = np.concatenate([hog_features, hsv_feat])
    return final_features.reshape(1, -1)


# =====================================================
#   PREPROCESSING & FEATURE EXTRACTION — MODEL NAUFAL
# =====================================================

def extract_color_features(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array([
        np.mean(img_rgb[:, :, 0]),
        np.mean(img_rgb[:, :, 1]),
        np.mean(img_rgb[:, :, 2])
    ])


def extract_features_naufal(img):
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feature = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )

    color_feat = extract_color_features(img)

    final_features = np.concatenate([hog_feature, color_feat])
    return final_features.reshape(1, -1)


# =====================================================
#                     ROUTES API
# =====================================================

@app.route("/predict_hayyin", methods=["POST"])
def predict_hayyin():
    if "image" not in request.files:
        return jsonify({"error": "Image not found"}), 400

    file = request.files["image"]
    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    features = extract_features_hayyin(img)
    pred = model_hayyin.predict(features)[0]

    return jsonify({
        "model": "hayyin",
        "prediction": classes_hayyin[pred]
    })


@app.route("/predict_naufal", methods=["POST"])
def predict_naufal():
    if "image" not in request.files:
        return jsonify({"error": "Image not found"}), 400

    file = request.files["image"]
    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    features = extract_features_naufal(img)
    pred = model_naufal.predict(features)[0]

    return jsonify({
        "model": "naufal",
        "prediction": classes_naufal[pred]
    })


# =====================================================
#                     RUN SERVER
# =====================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
