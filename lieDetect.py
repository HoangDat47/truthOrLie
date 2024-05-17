import tensorflow as tf
import json
import numpy as np
import cv2
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

# Tải mô hình từ file JSON và H5
with open('sequence_model.json', 'r') as json_file:
    model_json = json_file.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights('sequence_model.h5')

# Hàm xử lý video frames
def prepare_single_video(frames, MAX_SEQ_LENGTH, NUM_FEATURES):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i in range(len(frames)):
        if i >= MAX_SEQ_LENGTH:
            break  # Chỉ xử lý MAX_SEQ_LENGTH frame đầu tiên

        frame = cv2.resize(frames[i], (224, 224))  # Đảm bảo kích thước frame phù hợp với input của ResNet50
        frame = np.expand_dims(frame, axis=0)  # Thêm chiều batch
        frame = tf.keras.applications.resnet50.preprocess_input(frame)  # Tiền xử lý frame
        frame_features[0, i, :] = feature_extractor.predict(frame)  # Trích xuất đặc trưng từ frame
        frame_mask[0, i] = 1  # Đánh dấu frame không bị mask

    return frame_features, frame_mask


# Hàm load video
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frames[i], (224, 224))
        print(frames[i].shape)

        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    return frames

# Hàm kiểm tra nói dối
def check_lie(video_path, model, MAX_SEQ_LENGTH, NUM_FEATURES, class_vocab):
    frames = load_video(video_path)
    frame_features, frame_mask = prepare_single_video(frames, MAX_SEQ_LENGTH, NUM_FEATURES)

    probabilities = model.predict([frame_features, frame_mask])[0]

    # In kết quả
    for i in np.argsort(probabilities)[::-1]:
        print(f"{class_vocab[i]}: {probabilities[i] * 100:.2f}%")

    return probabilities

# Đường dẫn đến video thử nghiệm
test_video_path = 'lie.mp4'  # Điều chỉnh đường dẫn tới video của bạn

# Vocabulary lớp (điều chỉnh theo mô hình của bạn)
class_vocab = ['lie', 'truth']

# Kiểm tra nói dối
probabilities = check_lie(test_video_path, model, MAX_SEQ_LENGTH, NUM_FEATURES, class_vocab)

# In kết quả
for i in np.argsort(probabilities)[::-1]:
    print(f"{class_vocab[i]}: {probabilities[i] * 100:.2f}%")
