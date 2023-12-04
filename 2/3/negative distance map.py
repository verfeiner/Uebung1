
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def watershed():
    root=os.getcwd()#当前路径
    print(root)
    imagPath=os.path.join('testImages/pills.jpg')#添加路径
    img=cv.imread(imagPath)
    imagRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)#转彩色图
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)#转灰度图

    plt.figure()
    plt.subplot(331)    #nrows: 表示子图的行数。ncols: 表示子图的列数。index: 表示当前子图的索引，从左上角开始，从左到右，从上到下递增。
    plt.imshow(img,cmap='gray')

    plt.subplot(332)#注意数字间没有都逗号
    plt.imshow(imagRGB)#默认彩色是viridis

    plt.subplot(333) #二值化处理
    _,imagthreshold=cv.threshold(img,120,255,cv.THRESH_BINARY)
    plt.imshow(imagthreshold,cmap='gray')

    plt.subplot(334) #彭涨变换
    kernel=np.ones((3,3),np.uint8) #uint8 是一个数据类型的名称，表示无符号的8位整数。这个数据类型通常用于表示像素值或颜色通道的整数值，尤其在图像处理领域中很常见
    imgDilate=cv.morphologyEx(imagthreshold,cv.MORPH_ERODE,kernel)
    plt.imshow(imgDilate)

    plt.subplot(335) #距离图像
    distTrans=cv.distanceTransform(imagthreshold,cv.DIST_L2,3)#距离变换计算每个像素与其3x3邻域内像素, 较大的掩膜可以导致距离变换更加平滑，减少了噪声的影响。相反，较小的掩膜可能会更敏感于图像中的小细节。
    negative_dist_map = cv.normalize(-distTrans, None, 0, 1, cv.NORM_MINMAX)
    plt.imshow(negative_dist_map)

    plt.subplot(336) #距离图像二值化处理
    _,distThreshold=cv.threshold(negative_dist_map,0.6,1,cv.THRESH_BINARY)
    plt.imshow(distThreshold)

    plt.subplot(337)  #标记
    distThreshold=np.uint8(distThreshold) #标记
    _,labels=cv.connectedComponents(distThreshold)#uint8 (无符号8位整数)：取值范围： 0 到 255（2^8 - 1）。位数： 8位二进制，即一个字节。
    plt.imshow(labels)

    plt.figure()
    plt.subplot(121)
    labels=np.int32(labels)#int32 (有符号32位整数)：取值范围： -2147483648 到 2147483647（2^31 - 1 到 -(2^31)
    labels=cv.watershed(imagRGB,labels)
    plt.imshow(labels)

    plt.subplot(122)
    imagRGB[labels==-1]=[255,0,0]
    plt.imshow(imagRGB)



    plt.show()


if __name__=='__main__':
    watershed()