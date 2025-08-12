# STEP 1: Install required packages
!apt install tesseract-ocr -y
!pip install opencv-python pytesseract


import pytesseract
import cv2
import numpy as np
import matplotlib.pyplot as plt

# STEP 3: Create a test image with text
def create_test_image():
    image = np.ones((200, 600, 3), dtype=np.uint8) * 255  # White background
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Hello, this is a test!"
    cv2.putText(image, text, (50, 100), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return image

# STEP 4: OCR processing
def extract_text_from_image_array(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh)
    return text, image

# STEP 5: Generate image, run OCR
image = create_test_image()
text, display_image = extract_text_from_image_array(image)

# STEP 6: Display image and result
plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Generated Test Image")
plt.show()

print("📄 Extracted Text:")
print("-" * 40)
print(text.strip())
print("-" * 40)
