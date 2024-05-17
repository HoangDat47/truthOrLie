import cv2
from keras.models import model_from_json
import numpy as np

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Đường dẫn đến tệp video
video_path = "lie.mp4"
video = cv2.VideoCapture(video_path)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_counter = 0
label_counts = {label: 0 for label in labels.values()}

while True:
    ret, im = video.read()  # Đọc một frame từ video
    if not ret:
        break  # Kết thúc vòng lặp nếu không còn frame nào để đọc

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    try:
        if len(faces) > 0:  # Chỉ tính các frame mà đã phát hiện được khuôn mặt
            for (p, q, r, s) in faces:
                image = gray[q:q+s, p:p+r]
                cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                label_counts[prediction_label] += 1
                cv2.putText(im, f'{prediction_label}', (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

            frame_counter += 1
        cv2.imshow("Output", im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Thoát vòng lặp nếu nhấn 'q'
    except cv2.error:
        pass
# Hiển thị số lượng % mỗi nhãn
for label, count in label_counts.items():
    if frame_counter > 0:
        percentage = count/frame_counter*100
        print(f"{label}: {percentage:.2f}%")
    else:
        print(f"{label}: 0.00%")

# Hiển thị tổng phần trăm của tất cả các nhãn
if frame_counter > 0:
    total_percentage = sum(count/frame_counter*100 for count in label_counts.values())
    print(f"Total %: {total_percentage:.2f}%")
else:
    print("No face detected in any frame")

        
video.release()  # Giải phóng tài nguyên video
cv2.destroyAllWindows()  # Đóng cửa sổ khi kết thúc
