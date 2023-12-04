import cv2
import numpy as np

def dilation_reconstruction(original, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # 对每个通道分别进行腐蚀
    dilationed_image = cv2.dilate(original, kernel, iterations=1)

    # 对每个通道分别进行重建
    result = np.zeros_like(original)
    for i in range(3):
        original_channel = original[:, :, i]
        dilationed_channel = dilationed_image[:, :, i]
        marker = cv2.subtract(original_channel, dilationed_channel)
        result[:, :, i] = dilationed_channel+ cv2.dilate(marker, kernel, iterations=1)

    return result

def erosion_reconstruction(original,kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # 对每个通道分别进行腐蚀
    eroded_image = cv2.erode(original, kernel, iterations=1)

    # 对每个通道分别进行重建
    result = np.zeros_like(original)
    for i in range(3):
        original_channel = original[:, :, i]
        eroded_channel = eroded_image[:, :, i]
        marker = cv2.subtract(original_channel, eroded_channel)
        result[:, :, i] = eroded_channel - cv2.erode(marker, kernel, iterations=1)

    return result

def opening_reconstruction(image,kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    marker=cv2.erode(image,kernel)
    result=dilation_reconstruction(marker,kernel_size)
    return result
def closing_reconstruction(image,kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    marker=cv2.dilate(image,kernel)
    result=erosion_reconstruction(marker,kernel_size)
    return result
def smoothing_by_reconstruction(image,kernel_size):
    opened = opening_reconstruction(image, kernel_size)
    smoothed = closing_reconstruction(opened, kernel_size)
    return smoothed
def smoothing_reverse_by_reconstruction(image,kernel_size):
    closed = closing_reconstruction(image, kernel_size)
    smoothed = opening_reconstruction(closed, kernel_size)
    return smoothed
# 读取彩色图像
image = cv2.imread('C:/Users/Bin/Desktop/Master OBV/Master-WS/BALG/testImages/tools.jpg')

# 定义结构元素（这里使用一个矩形结构元素）
kernel_size = 5


# 彩色图像重建
reconstructed_image = dilation_reconstruction(image,kernel_size )
erosion_reconstructed_image=erosion_reconstruction(image,kernel_size)
opening_reconstruction_image=opening_reconstruction(image,kernel_size)
closing_reconstruction_image=closing_reconstruction(image,kernel_size)
smooth_reconstruction_image=smoothing_by_reconstruction(image,kernel_size)
smooth_reverse_reconstruction_image=smoothing_reverse_by_reconstruction(image,kernel_size)
# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('erosion', erosion_reconstructed_image)
cv2.imshow('dilation Image', reconstructed_image)
cv2.imshow('opening Image', opening_reconstruction_image)
cv2.imshow('closing Image', closing_reconstruction_image)
cv2.imshow('smooth image', smooth_reconstruction_image)
cv2.imshow('smooth reverse image', smooth_reverse_reconstruction_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
