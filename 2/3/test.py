import cv2
import numpy as np
from skimage import color, segmentation, filters, morphology
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('C:/Users/Bin/Desktop/Master OBV/Master-WS/BALG/testImages/pears.png')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算梯度
gradient = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, np.ones((3, 3)))

# 应用区域最大值和阈值标记法获取初步标记
markers = ndi.label(gradient > filters.threshold_otsu(gradient))[0]

# 使用Geodesic SKIZ获取前景和背景标记
edges = filters.sobel(gray_image)
geodesic = segmentation.inverse_gaussian_gradient(gray_image, edges, alpha=800, sigma=2)
geodesic_markers = segmentation.random_walker(geodesic, markers, beta=10, mode='bf')

# 分水岭变换
labels = segmentation.watershed(gradient, geodesic_markers, mask=gray_image)

# 提取每个苹果的区域
apple_regions = []
for label in np.unique(labels):
    if label == 0:
        continue  # 背景标签
    mask = np.zeros_like(gray_image, dtype=np.uint8)
    mask[labels == label] = 255
    apple_regions.append(mask)

# 可视化结果
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
ax = axes.ravel()

ax[0].imshow(rgb_image)
ax[0].set_title('原始图像')

ax[1].imshow(gradient, cmap=plt.cm.gray)
ax[1].set_title('梯度')

ax[2].imshow(geodesic_markers, cmap=plt.cm.nipy_spectral)
ax[2].set_title('Geodesic SKIZ 标记')

ax[3].imshow(labels, cmap=plt.cm.nipy_spectral)
ax[3].set_title('分水岭分割结果')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
