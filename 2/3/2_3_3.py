import cv2
import numpy as np
import matplotlib.pyplot as plt

def inverse_distance_transform(image):
    dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, inv_dist_transform = cv2.threshold(dist_transform, 0, 255, cv2.THRESH_BINARY_INV)
    return inv_dist_transform

def fill_holes(image):
    # 找到前景（亮）像素
    image = np.uint8(image)
    _, markers = cv2.connectedComponents(image)
    foreground = np.uint8(markers == markers[0, 0])

    # 创建一个以图像边界灰度值初始化的标记器
    marker = cv2.subtract(image, foreground)

    # 从标记器开始的侵蚀重建
    filled_holes = cv2.bitwise_or(image, cv2.bitwise_not(cv2.bitwise_not(marker) & cv2.bitwise_not(cv2.erode(marker, None))))

    return filled_holes

# 加载“pills”和“cells”图像
image_pills = cv2.imread('C:/Users/Bin/Desktop/Master OBV/Master-WS/BALG/testImages/pills.jpg', cv2.IMREAD_GRAYSCALE)
image_cells = cv2.imread('C:/Users/Bin/Desktop/Master OBV/Master-WS/BALG/testImages/img_cells.jpg', cv2.IMREAD_GRAYSCALE)

# 应用适当的阈值获得二值图像
_, binary_pills = cv2.threshold(image_pills, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

_, binary_cells = cv2.threshold(image_cells, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


# 应用反向距离变换
inv_dist_transform_pills = inverse_distance_transform(binary_pills)
inv_dist_transform_cells = inverse_distance_transform(binary_cells)

# 可选地对距离图进行平滑处理（例如，使用GaussianBlur）
inv_dist_transform_pills_smoothed = cv2.GaussianBlur(inv_dist_transform_pills, (0, 0), 3)
inv_dist_transform_cells_smoothed = cv2.GaussianBlur(inv_dist_transform_cells, (0, 0), 3)

# 可选地在反向距离图中填充孔洞
inv_dist_transform_pills_filled = fill_holes(inv_dist_transform_pills_smoothed)
inv_dist_transform_cells_filled = fill_holes(inv_dist_transform_cells_smoothed)

# 显示结果
plt.figure(figsize=(12, 6))

plt.subplot(231), plt.imshow(binary_pills, cmap='gray'), plt.title(' Pills')
plt.subplot(232), plt.imshow(inv_dist_transform_pills, cmap='gray'), plt.title('inverse distance Pills')
plt.subplot(233), plt.imshow(inv_dist_transform_pills_filled, cmap='gray'), plt.title('fill the hole Pills')

plt.subplot(234), plt.imshow(binary_cells, cmap='gray'), plt.title(' Cells')
plt.subplot(235), plt.imshow(inv_dist_transform_cells, cmap='gray'), plt.title('inverse distance Cells')
plt.subplot(236), plt.imshow(inv_dist_transform_cells_filled, cmap='gray'), plt.title('fill the hole Cells')

plt.tight_layout()
plt.show()
