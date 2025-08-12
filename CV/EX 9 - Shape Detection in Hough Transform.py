import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from IPython.display import display
from PIL import Image
import io

# Upload the file
uploaded = files.upload()

# Get the uploaded file path
for filename in uploaded.keys():
    image_path = filename

def detect_shapes(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from {image_path}.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Hough Circle Transform
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20,
                                param1=50, param2=30, minRadius=1, maxRadius=40)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 3:
            cv2.drawContours(image, [approx], 0, (0, 255, 255), 2)  # Triangle
        elif len(approx) == 4:
            cv2.drawContours(image, [approx], 0, (255, 0, 0), 2)    # Rectangle or square
        elif len(approx) > 4:
            if cv2.contourArea(contour) > 100:
                cv2.drawContours(image, [approx], 0, (255, 255, 0), 2)  # Circle-like

    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("Detected Shapes")
    plt.show()

# Run detection
detect_shapes(image_path)
