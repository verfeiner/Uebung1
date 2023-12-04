import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def reconstruction_by_dilation(marker, mask):
    result = np.zeros_like(marker)#大小为零，个数与目标一致
    while not np.array_equal(result, marker):
        result = np.copy(marker)
        marker = cv2.dilate(marker, np.ones((3, 3)))  # Structuring element: 3x3 square
        marker = np.minimum(marker, mask)
    return result

def reconstruction_by_erosion(marker, mask):
    result = np.zeros_like(marker)
    while not np.array_equal(result, marker):
        result = np.copy(marker)
        marker = cv2.erode(marker, np.ones((3, 3)))  # Structuring element: 3x3 square 全部是一的矩阵
        marker = np.maximum(marker, mask)
    return result

def opening_by_reconstruction(image, selem):
    marker = cv2.erode(image, selem)
    mask = image
    reconstruction_result = reconstruction_by_dilation(marker, mask)
    return reconstruction_result

def closing_by_reconstruction(image, selem):
    marker = cv2.dilate(image, selem)
    mask = image
    reconstruction_result = reconstruction_by_erosion(marker, mask)
    return reconstruction_result

def smoothing_by_reconstruction(image, selem_open, selem_close):
    opened = opening_by_reconstruction(image, selem_open)
    smoothed = closing_by_reconstruction(opened, selem_close)
    return smoothed

def smoothing_by_reconstruction_reverse(image, selem_open, selem_close):
    closed = closing_by_reconstruction(image, selem_close)
    smoothed = opening_by_reconstruction(closed, selem_open)
    return smoothed

# Example usage
# Create a sample binary image
image = cv2.imread('C:/Users/Bin/Desktop/Master OBV/Master-WS/BALG/testImages/tools.jpg')
#image[2:8, 2:8] = 1  # Add a square as an object

# Define structuring elements
selem=np.ones((7, 7), np.uint8)
selem_open = np.ones((7, 7), np.uint8)
selem_close = np.ones((7, 7), np.uint8)
t1=time.time()
result=opening_by_reconstruction(image,selem)
t2=time.time()
result2=closing_by_reconstruction(image,selem)
t3=time.time()

runtime_open=t2-t1
runtime_close=t3-t2
# Apply smoothing by reconstruction
t4=time.time()
smoothed = smoothing_by_reconstruction(image, selem_open, selem_close)
t5=time.time()
smoothed_reverse = smoothing_by_reconstruction_reverse(image, selem_open, selem_close)
t6=time.time()
runtime_smooth=t5-t4
runtime_smoothverse=t6-t5
print(f'runtime of opening ={runtime_open}')
print(f'runtime of closing ={runtime_close}')
print(f'runtime of smooth ={runtime_smooth}')
print(f'runtime of smoothverse ={runtime_smoothverse}')

# Display the results
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(image, cmap='gray', interpolation='none')
plt.title('Original Image')

plt.subplot(132)
plt.imshow(result, cmap='gray', interpolation='none')
plt.title('opening by Reconstruction')

plt.subplot(133)
plt.imshow(result2, cmap='gray', interpolation='none')
plt.title('closing by Reconstruction (Reverse)')

plt.show()
# Display the results
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(image, cmap='gray', interpolation='none')
plt.title('Original Image')

plt.subplot(132)
plt.imshow(smoothed, cmap='gray', interpolation='none')
plt.title('Smoothing by Reconstruction')

plt.subplot(133)
plt.imshow(smoothed_reverse, cmap='gray', interpolation='none')
plt.title('Smoothing by Reconstruction (Reverse)')

plt.show()
