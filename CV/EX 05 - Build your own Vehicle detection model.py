# Install YOLOv5 dependencies
!git clone https://github.com/ultralytics/yolov5  # Clone repo
%cd yolov5
!pip install -r requirements.txt


# Step 2: Upload your car image
from google.colab import files
uploaded = files.upload()  # Upload your image, e.g., cars.jpeg


import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load YOLOv5 pretrained model (trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Perform inference
results = model('cars.png')  # Replace with your file name

# Show detected image
results.show()

# Print detected classes
results.print()
