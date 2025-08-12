import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files
from io import BytesIO
from PIL import Image

# Upload the image file
uploaded = files.upload()

# Get the uploaded file name
file_name = next(iter(uploaded))

# Read the image using OpenCV
file_bytes = np.asarray(bytearray(uploaded[file_name]), dtype=np.uint8)
image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
output = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a median blur to reduce noise
gray_blurred = cv2.medianBlur(gray, 5)

# Detect circles using Hough Transform
circles = cv2.HoughCircles(
    gray_blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=100,
    param2=30,
    minRadius=0,
    maxRadius=0
)

# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    print("Detected Circles:")
    for (x, y, r) in circles[0, :]:
        print(f"Center: ({x}, {y}), Radius: {r}")
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # outer circle
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # center point
else:
    print("No circles detected.")

# Show the result
cv2_imshow(output)
