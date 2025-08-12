# Step 1: Upload the image
from google.colab import files
uploaded = files.upload()  # Choose your image file

# Step 2: Import libraries
import cv2
from google.colab.patches import cv2_imshow
import numpy as np

# Step 3: Read the uploaded image
image = cv2.imread('im1.jpg')  # Replace with actual uploaded filename if different

# Step 4: Check if image loaded properly
if image is None:
    print("Error: Image not loaded. Check filename.")
else:
    # Resize the image to 200x282
    image = cv2.resize(image, (282, 200))  # width x height

    # Display original resized image
    print("Original Resized Image (200x282):")
    cv2_imshow(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Grayscale Image:")
    cv2_imshow(gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    print("Blurred Image:")
    cv2_imshow(blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Canny Edge Detection
    edges = cv2.Canny(gray_image, 100, 200)
    print("Canny Edges:")
    cv2_imshow(edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Draw a smaller rectangle and a line inside
    cv2.rectangle(image, (100, 60), (180, 140), (255, 0, 0), 2)  # Blue rectangle
    cv2.line(image, (100, 60), (180, 140), (0, 0, 255), 2)       # Red line

    print("Image with Smaller Rectangle and Line:")
    cv2_imshow(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
