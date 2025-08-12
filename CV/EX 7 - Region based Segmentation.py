import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage import data, segmentation, morphology, filters
from skimage.color import rgb2gray, label2rgb
import scipy.ndimage as nd

plt.rcParams["figure.figsize"] = (12, 8)

# Load image and convert to grayscale
rocket = data.rocket()
rocket_wh = rgb2gray(rocket)

# Apply Canny edge detection
edges = canny(rocket_wh)
plt.imshow(edges, interpolation='gaussian')
plt.title('Canny detector')
plt.show()

# Fill regions to perform edge segmentation
fill_im = nd.binary_fill_holes(edges)
plt.imshow(fill_im)
plt.title('Region Filling')
plt.show()

# Compute the elevation map using the Sobel filter
elevation_map = filters.sobel(rocket_wh)
plt.imshow(elevation_map)
plt.title('Elevation Map')
plt.show()

# Create markers for watershed
markers = np.zeros_like(rocket_wh, dtype=np.int32)
markers[rocket_wh < 30/255] = 1
markers[rocket_wh > 150/255] = 2
plt.imshow(markers)
plt.title('Markers')
plt.show()

# Perform watershed segmentation
segments = segmentation.watershed(elevation_map, markers)
plt.imshow(segments)
plt.title('Watershed Segmentation')
plt.show()

# Fill holes in the segmented image
segments_filled = nd.binary_fill_holes(segments - 1)
label_rock, _ = nd.label(segments_filled)

# Overlay image with different labels
image_label_overlay = label2rgb(label_rock, image=rocket_wh)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16), sharey=True)
ax1.imshow(rocket_wh)
ax1.contour(segments_filled, [0.8], linewidths=1.8, colors='w')
ax2.imshow(image_label_overlay)
plt.show()
