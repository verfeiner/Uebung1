import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import skimage as ski
import scipy
from scipy import ndimage

def gradientmap():
    root=os.getcwd()
    imagepath=os.path.join('testImages/pears.png')
    img=cv2.imread(imagepath)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    plt.figure()
    plt.subplot(331)
    plt.imshow(img,cmap='gray')

    #形态学梯度对于突出图像中物体的边界或边缘非常有用。它经常用作目标检测和分割等任务的预处理步骤。G=dilate(I)−erode(I)
    imggauss=ski.filters.gaussian(img)
    gradient=cv2.morphologyEx(imggauss,cv2.MORPH_GRADIENT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))#椭圆形状的结构元素具有方向性，可以更好地捕捉图像中的椭圆状结构
    plt.subplot(332)
    plt.imshow(gradient)

    #gradient后的分水岭虽然相比原图有了边界，但是过度分线
    gradeint_water=ski.segmentation.watershed(gradient,watershed_line=True)#trueT必须大写
    plt.subplot(333)
    plt.imshow(gradeint_water)

    imgerosion=cv2.erode(img,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)),iterations=1)
    imgreco=ski.morphology.reconstruction(imgerosion,img,method='dilation')

    imgdilation=cv2.dilate(imgreco,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)),iterations=1)
    imgreco=ski.morphology.reconstruction(imgdilation,imgreco,method='erosion')

    imgdilation=cv2.dilate(imgreco,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)),iterations=1)
    imgreco=ski.morphology.reconstruction(imgdilation,imgreco,method='erosion')

    imgerosion = cv2.erode(imgreco, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21, 21)), iterations=1)
    imgreco = ski.morphology.reconstruction(imgerosion, imgreco, method='dilation')
    plt.subplot(334)
    plt.imshow(imgreco,cmap='gray')

    local_max=ski.feature.peak_local_max(imgreco,min_distance=60)
    fg_lokal_max=np.zeros_like(imgreco)
    fg_lokal_max[tuple(local_max.T)]=255#，灰度值255通常表示白色
    plt.subplot(335)
    plt.imshow(fg_lokal_max)

    _,fg_threshold=cv2.threshold(imgreco,140,250,cv2.THRESH_BINARY)#不要忘记，分threshold算法输出两个量
    fg_threshold=cv2.erode(fg_threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    fg_threshold=cv2.medianBlur(fg_threshold.astype(np.uint8),5)
    #fg_threshold=ski.filters.gaussian(fg_threshold)
    plt.subplot(336)
    plt.imshow(fg_threshold)

    inv_fg = np.zeros_like(fg_threshold)
    inv_fg[fg_threshold == 0] = 255
    dist = cv2.distanceTransform(inv_fg.astype(np.uint8), cv2.DIST_L2, 3)
    labels_fg_threshold = ndimage.label(fg_threshold)[0]


    skiz = ski.segmentation.watershed(dist, labels_fg_threshold, watershed_line=True)
    skiz_lines = np.zeros_like(skiz)  # Es werden nur die Watershed-Linien benötigt
    skiz_lines[skiz == 0] = 255

    plt.subplot(337)
    plt.imshow(dist,cmap='gray')

    plt.subplot(338)
    plt.imshow(skiz_lines, cmap='gray')

    marker_fg_local_max = ndimage.label(fg_lokal_max)[0]
    marker_fg_threshold = ndimage.label(fg_threshold)[0]



    watershed_fg_local_max = ski.segmentation.watershed(gradient, marker_fg_local_max, watershed_line=True)  # Watershed ohne Background
    watershed_fg_threshold = ski.segmentation.watershed(gradient, marker_fg_threshold, watershed_line=True)

    plt.figure()
    plt.subplot(231)
    plt.imshow(watershed_fg_local_max)
    plt.subplot(232)
    plt.imshow(watershed_fg_threshold)
    labels_fg_bg_localmax = marker_fg_local_max + 1  # Alle Marker werden um 1 erhöht, sodass das Markerlabel 0 frei/unbenutzt ist
    flood_area_local_max = fg_lokal_max.astype(np.uint8) + skiz_lines
    labels_fg_bg_localmax[flood_area_local_max==0]=0

    labels_fg_bg_threshold = marker_fg_threshold+1
    flood_area_threshild=fg_threshold.astype(np.uint8) + skiz_lines
    labels_fg_bg_threshold[flood_area_threshild == 0] = 0


    #labels_fg_bg_localmax=cv2.watershed(gradient,labels_fg_bg_localmax)
    # labels_fg_bg_threshold=cv2.watershed(gradient,labels_fg_bg_threshold)

    watershed_fg_bg_local_max = ski.segmentation.watershed(gradient, labels_fg_bg_localmax,watershed_line=True)  # Watershed ohne Background
    watershed_fg_bg_threshold = ski.segmentation.watershed(gradient, labels_fg_bg_threshold, watershed_line=True)

    watershed_fg_bg_local_max=np.int32(watershed_fg_bg_local_max)
    watershed_fg_bg_threshold=np.int32(watershed_fg_bg_threshold)
    imgRGB[watershed_fg_bg_threshold == 0] = [255, 0, 0]#重点ski的segementation变换后分割线的标记不变还是0

    plt.subplot(233)
    plt.imshow(imgRGB)

    plt.subplot(235)
    plt.imshow(watershed_fg_bg_local_max,cmap='gray')

    plt.subplot(234)
    plt.imshow(flood_area_threshild,cmap='gray')



    plt.show()

if __name__=='__main__':
    gradientmap()