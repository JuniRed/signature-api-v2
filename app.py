from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

def decode_base64_to_image(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        img_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(img_data)).convert("L")  # Convert to grayscale
        return np.array(img)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def preprocess_image(img):
    try:
        # Resize and denoise
        img = cv2.resize(img, (400, 200))
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Improve contrast with adaptive thresholding
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        return img
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return img

def compare_signatures_orb(img1, img2):
    try:
        img1 = preprocess_image(img1)
        img2 = preprocess_image(img2)

        orb = cv2.ORB_create(nfeatures=1000)

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        print(f"Keypoints img1: {len(kp1) if kp1 else 0}, img2: {len(kp2) if kp2 else 0}")

        if des1 is None or des2 is None:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        good_matches = [m for m in matches if m.distance < 60]

        match_ratio = len(good_matches) / max(len(kp1), len(kp2))
        similarity = round(match_ratio * 100, 2)

        return similarity
    except Exception as e:
        print(f"Error comparing signatures with ORB: {e}")
        return 0.0

@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()

    document_base64 = data.get("image1", "")
    reference_base64 = data.get("image2", "")

    if not document_base64 or not reference_base64:
        return jsonify({"error": "Both images are required"}), 400

    document_img = decode_base64_to_image(document_base64)
    reference_img = decode_base64_to_image(reference_base64)

    if document_img is None or reference_img is None:
        return jsonify({"error": "Invalid image data"}), 400

    similarity = compare_signatures_orb(document_img, reference_img)

    return jsonify({
        "similarity": similarity,
        "match": similarity >= 40
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
