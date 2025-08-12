# STEP 1: Clone YOLOv5 and install dependencies
!pip install ultralytics opencv-python


import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from google.colab import files

# Upload video
uploaded = files.upload()
video_path = list(uploaded.keys())[0]

# Load model
model = YOLO('yolov5s.pt')

# Read video and extract last frame
cap = cv2.VideoCapture(video_path)
last_frame = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    last_frame = frame
cap.release()

if last_frame is None:
    raise ValueError("No valid frame found in the video.")

# Detect
results = model(last_frame)[0]

# Filtering thresholds
CONF_THRESH = 0.4
AREA_THRESH = 5000  # pixels (adjust as needed)

person_count = 0
for i, box in enumerate(results.boxes):
    cls_id = int(box.cls[0])
    label = model.names[cls_id]
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    area = (x2 - x1) * (y2 - y1)

    # Only draw box if it's a person and meets size + confidence filters
    if label == 'person' and conf > CONF_THRESH and area > AREA_THRESH:
        person_count += 1
        cv2.rectangle(last_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(last_frame, f"Person {person_count}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Display total count
cv2.putText(last_frame, f"Total Persons: {person_count}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Show final result
last_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(last_rgb)
plt.axis('off')
plt.title("Filtered Final Frame with People Count")
plt.show()
