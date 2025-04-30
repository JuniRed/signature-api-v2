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
        # Remove header if present (e.g., data:image/png;base64,...)
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        img_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(img_data)).convert("L")  # grayscale
        return np.array(img)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def compare_images(img1, img2):
    try:
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        diff = cv2.absdiff(img1, img2)
        non_zero_count = np.count_nonzero(diff)
        total_pixels = img1.size
        similarity = 100 - (non_zero_count / total_pixels * 100)
        return round(similarity, 2)
    except Exception as e:
        print(f"Error comparing images: {e}")
        return 0.0

@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()

    image1_base64 = data.get("image1", "")
    image2_base64 = data.get("image2", "")

    if not image1_base64 or not image2_base64:
        return jsonify({"error": "Both images are required"}), 400

    img1 = decode_base64_to_image(image1_base64)
    img2 = decode_base64_to_image(image2_base64)

    if img1 is None or img2 is None:
        return jsonify({"error": "Invalid image data"}), 400

    similarity = compare_images(img1, img2)

    return jsonify({"similarity": similarity}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # For Render
    app.run(host='0.0.0.0', port=port)
