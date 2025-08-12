!pip install ultralytics


from google.colab import files
from google.colab.patches import cv2_imshow
import cv2
import numpy as np
from scipy.spatial import distance
from itertools import combinations
from ultralytics import YOLO

uploaded = files.upload()
video_path = list(uploaded.keys())[0]

model = YOLO("yolov8n.pt")

def get_centroid(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def is_violating(c1, c2, threshold=75):
    return distance.euclidean(c1, c2) < threshold

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_count > 100:  # Limit to first 100 frames to avoid long output
        break

    results = model(frame, verbose=False)[0]
    person_boxes = []
    for r in results.boxes:
        if int(r.cls[0]) == 0:  # class 0 is person
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            person_boxes.append((x1, y1, x2, y2))

    centroids = [get_centroid(b) for b in person_boxes]
    violations = set()
    for (i, c1), (j, c2) in combinations(enumerate(centroids), 2):
        if is_violating(c1, c2):
            violations.add(i)
            violations.add(j)
            cv2.line(frame, c1, c2, (0, 0, 255), 2)

    for i, box in enumerate(person_boxes):
        x1, y1, x2, y2 = box
        cx, cy = get_centroid(box)
        color = (0, 255, 0) if i not in violations else (0, 0, 255)
        label = "Safe" if i not in violations else "Violation"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, (cx, cy), 5, color, -1)

    cv2_imshow(frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
