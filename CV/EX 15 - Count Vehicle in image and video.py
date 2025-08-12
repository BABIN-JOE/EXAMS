# Install YOLOv5 dependencies
!git clone https://github.com/ultralytics/yolov5  # Clone repo
%cd yolov5
!pip install -r requirements.txt


from google.colab import files
uploaded = files.upload()  # Upload your image, e.g., cars.jpg

import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load YOLOv5 pretrained model (trained on COCO)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Get uploaded filename
img_path = list(uploaded.keys())[0]

# Run inference
results = model(img_path)

# Filter detections for cars only (COCO class id for 'car' is 2)
car_detections = results.pred[0][results.pred[0][:, -1] == 2]

car_count = car_detections.shape[0]

print(f"Number of cars detected: {car_count}")

# Display image with detections and counts as title
results.show()

plt.title(f'Cars detected: {car_count}')
plt.show()
