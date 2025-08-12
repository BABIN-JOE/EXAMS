!pip install deepface


import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from google.colab import files

uploaded = files.upload()
image_path = list(uploaded.keys())[0]

img = cv2.imread(image_path)
plt.imshow(img[:, :, ::-1])  # BGR to RGB for correct colors
plt.axis('off')
plt.show()

result = DeepFace.analyze(img, actions=['emotion'])
print(result)
