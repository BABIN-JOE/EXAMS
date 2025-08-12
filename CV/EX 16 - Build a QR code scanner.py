!apt-get install -y libzbar0
!pip install pyzbar


import cv2
import numpy as np
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt
from google.colab import files

def decode_qr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    qr_codes = decode(gray)
    return qr_codes

uploaded = files.upload()  # Upload your image file

image_path = list(uploaded.keys())[0]  # Get uploaded file name
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not read the image file")
else:
    qr_codes = decode_qr(frame)
    for qr_code in qr_codes:
        qr_data = qr_code.data.decode('utf-8')
        points = qr_code.polygon

        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            cv2.polylines(frame, [hull.astype(np.int32)], True, (255, 0, 0), 3)
        else:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], True, (255, 0, 0), 3)

        print("QR Code detected:", qr_data)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show()
