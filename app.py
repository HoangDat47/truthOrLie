from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
import json
import random

# TensorFlow
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_seed()

# Define a Flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
MODEL_PATH = 'sequence_model.h5'

# Load model
json_file = open("sequence_model.json", "r")
model_json = json_file.read()
json_file.close()
sequence_model = model_from_json(model_json)
sequence_model.load_weights("sequence_model.h5")

class_vocab = ["lie", "truth"]

print('Model loaded. Check http://127.0.0.1:5000/')

# Create feature extractor with InceptionV3
base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_extractor = tf.keras.Sequential([
    base_model,
    global_average_layer
])

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def sequence_prediction(path):
    frames = load_video(path)  # Đọc video
    frame_features, frame_mask = prepare_single_video(frames)  # Chuẩn bị dữ liệu cho mô hình
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]  # Dự đoán

    results = {class_vocab[0]: probabilities[0] * 100, class_vocab[1]: probabilities[1] * 100}
    return results

def load_video(video_path, max_frames=MAX_SEQ_LENGTH):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))  # Đảm bảo kích thước frame phù hợp với input của mô hình
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    return frames

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to a temporary location
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = sequence_prediction(file_path)

        # Remove the file after prediction
        os.remove(file_path)

        return jsonify(result)
    return None

if __name__ == '__main__':
    app.run(debug=True)
