!pip install opencv-python-headless


!wget https://github.com/opencv/opencv/raw/master/samples/data/vtest.avi -O traffic.avi


video_path = 'traffic.avi'


import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path = 'traffic.avi'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Couldn't open video.")
else:
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    dot_locations = []
    last_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame.copy()
        fg_mask = cv2.medianBlur(bg_subtractor.apply(frame), 5)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 100:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dot_locations.append((cx, cy))

    cap.release()

    # Draw all accumulated blue dots on the last frame
    for (cx, cy) in dot_locations:
        cv2.circle(last_frame, (cx, cy), 3, (255, 0, 0), -1)  # Blue dot

    # Convert to RGB and show
    final_img = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 7))
    plt.imshow(final_img)
    plt.title("Final Motion Detection with Blue Dots")
    plt.axis('off')
    plt.show()

