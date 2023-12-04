import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.morphology import dilation, disk
from skimage import img_as_ubyte



def naive_reconstruction_by_dilation(marker, mask, kernel, max_iterations=100):
    result = np.copy(marker)
    tolerance = 1e-5

    for _ in range(max_iterations):
        dilated = np.maximum(result, convolve(result, kernel, mode='constant', cval=0.0))
        new_result = np.minimum(dilated, mask)

        if np.all(np.abs(result - new_result) < tolerance):
            break

        result = new_result

    return result



def reconstruction_by_dilation(marker, mask, selem, max_iterations=100):
    result = marker.copy()

    for _ in range(max_iterations):
        dilated = dilation(result, selem)
        result = np.minimum(dilated, mask)

        if np.array_equal(result, dilated): #如果两个数组具有相同的形状和元素，则为 True，否则为 False。
            break

    return result


# Example usage
# Create a sample binary image 点重合也算膨胀
image = np.zeros((20, 30), dtype=np.uint8)
image[2:6, 2:6] = 1  # Add a square as a marker

# Create a mask with a larger square
mask = np.zeros((20, 30), dtype=np.uint8)
mask[1:8, 1:8] = 1

# Define a structuring element (kernel)
kernel = np.ones((3, 3), np.uint8)
selem=disk(3)

# Apply naive reconstruction by dilation
t1=time.time()
result = naive_reconstruction_by_dilation(image, mask, kernel)
t2=time.time()
laufzeit_normal=t2-t1
t3=time.time()
result2=reconstruction_by_dilation(image, mask, selem)
t4=time.time()
Laufzeit_scikit=t4-t3
# Display the results
print(f'runtime of reconstruction by dilation={laufzeit_normal}')
print(f'runtime of scikitimage={Laufzeit_scikit}')

plt.figure(figsize=(10, 4))

plt.subplot(131)
plt.imshow(image, cmap='gray', interpolation='none')
plt.title('Marker')

plt.subplot(132)
plt.imshow(result2, cmap='gray', interpolation='none')
plt.title('Reconstruction by scikit-image')

plt.subplot(133)
plt.imshow(result, cmap='gray', interpolation='none')
plt.title('Reconstruction by Dilation')

plt.show()
