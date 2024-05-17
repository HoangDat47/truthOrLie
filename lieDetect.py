import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import json
import numpy as np
import cv2
import pandas as pd

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# Đọc mô hình từ file JSON
json_file = open("sequence_model.json", "r")
model_json = json_file.read()
json_file.close()
sequence_model = model_from_json(model_json)

# Load trọng số cho sequence_model từ file "sequence_model.h5"
sequence_model.load_weights("sequence_model.h5")

# Danh sách lớp
class_vocab = ["lie", "truth"]

# Hàm chuẩn bị dữ liệu cho mỗi video
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

# Hàm dự đoán lớp của video
def sequence_prediction(path):
    frames = load_video(path)  # Đọc video
    frame_features, frame_mask = prepare_single_video(frames)  # Chuẩn bị dữ liệu cho mô hình
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]  # Dự đoán

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames

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


# Danh sách video cần kiểm tra
test_df = pd.DataFrame({
    "video_name": ["lie.mp4", "true.mp4"]  # Thay đổi danh sách video cần kiểm tra tùy thuộc vào dữ liệu thực tế
})

# Chọn một video ngẫu nhiên từ danh sách
test_video = np.random.choice(test_df["video_name"].values.tolist())
print(f"Test video path: {test_video}")

# Kiểm tra video đã chọn
test_frames = sequence_prediction(test_video)
