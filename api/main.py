from flask import Flask, request, jsonify
from flask_cors import CORS
from keras_facenet import FaceNet
from PIL import Image
import os
import io
from datetime import datetime
import logging
import json
import numpy as np
import pickle

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ImageReceiver')

app = Flask(__name__)

CORS(app)


# Initialize FaceNet (will produce 128D embeddings) 
embedder = None

def get_embedder():
    global embedder
    if embedder is None:
        logging.info("Loading FaceNet model...")
        embedder = FaceNet()
        # Verify embedding dimension
        dummy_input = np.zeros((1, 160, 160, 3))  # FaceNet's expected input shape
        dummy_embedding = embedder.embeddings(dummy_input)[0]
        logging.info(f"FaceNet embedding dimension: {len(dummy_embedding)}")
    return embedder


with open("./api/database.pkl", "rb") as file:
    database = pickle.load(file)

@app.route("/")
def home():
    return "HELLO WORLD"

@app.route('/reconocer_persona', methods=['POST'])
def reconocer_persona():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty file'}), 400
        
        # Read and preprocess image
        img = Image.open(io.BytesIO(file.read()))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to FaceNet's expected input (160x160)
        img = img.resize((160, 160))
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        
        # Get FaceNet embeddings (will be 128D)
        embedder = get_embedder()
        detections = embedder.embeddings([img_array])
        
        if len(detections) == 0:
            return jsonify({'error': 'No faces detected'}), 400
        
        # Verify embedding dimension
        embedding = detections[0]
        if len(embedding) != 128:
            logging.warning(f"Unexpected embedding dimension: {len(embedding)}")
        
        return jsonify({
            'embedding': embedding.tolist(),  # 128D vector
            'dimension': len(embedding),
            'face_count': len(detections),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ----------------------------------------------------------------------------------

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)