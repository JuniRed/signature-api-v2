import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

def decode_base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string.split(",")[-1])
    img = Image.open(BytesIO(img_data)).convert("L")
    return np.array(img)

def compare_images(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    diff = cv2.absdiff(img1, img2)
    non_zero_count = np.count_nonzero(diff)
    total_pixels = img1.size
    similarity = 100 - (non_zero_count / total_pixels * 100)
    return round(similarity, 2)

@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    img1 = decode_base64_to_image(data.get("image1"))
    img2 = decode_base64_to_image(data.get("image2"))

    if img1 is None or img2 is None:
        return jsonify({"error": "Invalid image data"}), 400

    score = compare_images(img1, img2)
    return jsonify({"similarity": score})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # <- Render sets this
    app.run(host='0.0.0.0', port=port)
