import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload the image
uploaded = files.upload()
file_name = next(iter(uploaded))

# Read image from uploaded file
file_bytes = np.asarray(bytearray(uploaded[file_name]), dtype=np.uint8)
image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur and detect edges
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
edges = cv2.Canny(blurred, 30, 100)

# Optional dilation to enhance lines
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)

# Detect lines using Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)

# Draw detected lines
if lines is not None:
    print("Detected Lines:")
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
else:
    print("No lines detected.")

# Show the results
print("Edges:")
cv2_imshow(edges)

print("Detected Lines on Image:")
cv2_imshow(image)
