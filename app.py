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
        img = Image.open(BytesIO(img_data)).convert("L")  # grayscale
        return np.array(img)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def extract_signature_region(image):
    try:
        _, thresh = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            signature = image[y:y+h, x:x+w]
            return signature
        return None
    except Exception as e:
        print(f"Error extracting signature: {e}")
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

    document_base64 = data.get("image1", "")
    reference_base64 = data.get("image2", "")

    if not document_base64 or not reference_base64:
        return jsonify({"error": "Both document_image and reference_signature are required"}), 400

    document_img = decode_base64_to_image(document_base64)
    reference_img = decode_base64_to_image(reference_base64)

    if document_img is None or reference_img is None:
        return jsonify({"error": "Invalid image data"}), 400

    signature_region = extract_signature_region(document_img)

    if signature_region is None:
        return jsonify({"error": "No signature found in document image"}), 400

    similarity = compare_images(signature_region, reference_img)

    return jsonify({
        "similarity": similarity,
        "match": similarity >= 80  # threshold can be adjusted
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # For Render/Railway
    app.run(host='0.0.0.0', port=port)
