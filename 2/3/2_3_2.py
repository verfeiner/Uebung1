
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Generate a blank image
image_size = (300, 300)
image = np.zeros(image_size, dtype=np.uint8)

# Define the parameters for three circles
circle1 = (100, 100, 50)
circle2 = (180, 120, 40)
circle3 = (60, 200, 35)

# Draw three circles on the image
cv2.circle(image, (circle1[0], circle1[1]), circle1[2], 255, -1)
cv2.circle(image, (circle2[0], circle2[1]), circle2[2], 255, -1)
cv2.circle(image, (circle3[0], circle3[1]), circle3[2], 255, -1)

# Display the generated image
plt.imshow(image, cmap='gray')
plt.title('Generated Binary Image with Circles')
plt.show()

# Calculate the negative distance map
dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_5)
_, sure_fg = cv2.threshold(dist_transform, 0.5* dist_transform.max(), 255, 0)

cv2.imshow('dist_transform',dist_transform)
cv2.imshow('sure_fg',sure_fg)
# Finding background area
sure_bg = cv2.dilate(image, None, iterations=3)
cv2.imshow('sure_bg',sure_bg)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
cv2.imshow('unknown',unknown)

# Marking the labels
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Applying watershed transform
cv2.watershed(cv2.cvtColor(cv2.merge([image, image, image]), cv2.COLOR_BGR2RGB), markers)

# Display the segmented image
plt.imshow(markers, cmap='nipy_spectral')
plt.title('Segmented Image using Watershed Transform')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()