
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io
import ipywidgets as widgets
from IPython.display import display

def handle_upload(change):
    file = next(iter(uploader.value.values()))
    img_data = file['content']
    img = Image.open(io.BytesIO(img_data)).convert("L")
    process_image(np.array(img))

def process_image(gray):
    # Resize for faster computation
    gray = cv2.resize(gray, (256, 256))

    # Apply Gaussian Blur to smooth
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Generate "depth" using Sobel filters
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    depth = np.sqrt(grad_x**2 + grad_y**2)
    depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

    # Create 3D plot
    X, Y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, depth, cmap='plasma', edgecolor='none')
    ax.set_title('3D Shape Estimation from 2D Image')
    plt.show()

# Upload widget
uploader = widgets.FileUpload(accept='image/*', multiple=False)
uploader.observe(handle_upload, names='value')
display(uploader)
