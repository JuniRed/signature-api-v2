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

def extract_signature_region(image):
    try:
        height, width = image.shape
        crop_y_start = int(height * 0.75)
        cropped = image[crop_y_start:height, 0:width]

        _, thresh = cv2.threshold(cropped, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            return cropped[y:y+h, x:x+w]
        else:
            return cropped
    except Exception as e:
        print(f"Error extracting signature: {e}")
        return None

def compare_signatures_orb(img1, img2):
    try:
        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Calculate similarity score based on good matches
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 50]

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
        return jsonify({"error": "Both document_image and reference_signature are required"}), 400

    document_img = decode_base64_to_image(document_base64)
    reference_img = decode_base64_to_image(reference_base64)

    if document_img is None or reference_img is None:
        return jsonify({"error": "Invalid image data"}), 400

    signature_region = extract_signature_region(document_img)

    if signature_region is None:
        return jsonify({"error": "No signature found in document image"}), 400

    similarity = compare_signatures_orb(signature_region, reference_img)

    return jsonify({
        "similarity": similarity,
        "match": similarity >= 40  # ORB match threshold
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
