from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import json

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048
MODEL_PATH = 'sequence_model.h5'

json_file = open("sequence_model.json", "r")
model_json = json_file.read()
json_file.close()
sequence_model = model_from_json(model_json)
class_vocab = ["lie", "truth"]

print('Model loaded. Check http://127.0.0.1:5000/')

# Tạo một mô hình ResNet50 không bao gồm lớp đầu ra (include_top=False)
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

# Tạo một lớp global average pooling layer để chuyển đổi đầu ra của ResNet50 thành đặc trưng
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# Kết hợp ResNet50 và lớp global average pooling layer để tạo ra một feature extractor
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

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
        result = (f"  {class_vocab[0]}: {probabilities[0] * 100:5.2f}% and {class_vocab[1]}: {probabilities[1] * 100:5.2f}%")
    return result

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
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


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Ensure the uploads directory exists
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)

        # Save the file to ./uploads
        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = sequence_prediction(file_path)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
