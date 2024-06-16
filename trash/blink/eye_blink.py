import os
import cv2
import dlib
import numpy as np
import pickle

# Hàm tính toán tỷ lệ chớp mắt từ mắt
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Tải mô hình nhận diện khuôn mặt và landmark
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.isfile(predictor_path):
    raise FileNotFoundError(f"File not found: {predictor_path}")
predictor = dlib.shape_predictor(predictor_path)

# Chỉ số của mắt trái và phải trên landmark
(left_eye_start, left_eye_end) = (42, 48)
(right_eye_start, right_eye_end) = (36, 42)

# Đường dẫn đến video
video_path = "true.mp4"

print(f"Processing video: {video_path}")
cap = cv2.VideoCapture(video_path)
blink_count = 0
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        left_eye = shape[left_eye_start:left_eye_end]
        right_eye = shape[right_eye_start:right_eye_end]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        if ear < 0.25:  # Ngưỡng cho tỷ lệ chớp mắt
            blink_count += 1

blink_ratio = blink_count / frame_count if frame_count > 0 else 0
print(f"Blink ratio: {blink_ratio}")

# Tính phần trăm nói dối và phần trăm nói thật
truth_percentage = blink_ratio * 100 if blink_ratio <= 0.85 else 85
lie_percentage = (1 - blink_ratio) * 100 if blink_ratio >= 0.15 else 15

print(f"Phần trăm nói dối: {lie_percentage}%")
print(f"Phần trăm nói thật: {truth_percentage}%")

# Đọc mô hình từ file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Chuẩn bị dữ liệu cho mô hình
# Ví dụ: X là tỷ lệ chớp mắt, bạn cần chuyển đổi nó thành một định dạng phù hợp để đưa vào mô hình

# Dự đoán bằng mô hình
predicted_label = model.predict(X)

print("Predicted label:", predicted_label)

cap.release()
