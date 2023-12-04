import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/Bin/Desktop/Master OBV/Master-WS/BALG/testImages/coin.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)# Otsu's 二值化对图像进行二值化
cv2.imshow('tresh',thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2) #开运算的效果包括去除小物体、平滑边缘和断开物体之间的细小连接
cv2.imshow('opening',opening)

sure_bg = cv2.dilate(opening, kernel, iterations=2)  # sure background area

cv2.imshow('sure_bg ',sure_bg )

sure_fg = cv2.erode(opening, kernel, iterations=2)  # sure foreground area
cv2.imshow('sure_fg ',sure_fg )

unknown = cv2.subtract(sure_bg, sure_fg)  # unknown area
cv2.imshow('unknown ',unknown  )
# Perform the distance transform algorithm
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)# cv2.DIST_L2 简单欧几里得距离（欧氏距离）

cv2.imshow('dist_transform ',dist_transform  )
# Normalize the distance image for range = {0.0, 1.0}
cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)#当图像内的各个子图没有连接时：可以直接使用形态学的腐蚀操作确定前景对象。但是当图像内的子图连接在一起时：就需要借助距离变换函数提取前景对象。

# Finding sure foreground area
ret, sure_fg2 = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)#对其进行二值处理就得到了分离的前景图（下面中间的图）

# Finding unknown region
sure_fg2 = np.uint8(sure_fg2)
unknown2 = cv2.subtract(sure_bg,sure_fg2)
cv2.imshow('sure_fg2',sure_fg2)
cv2.imshow( 'unknown2  ',unknown2  )

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg2)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown2==255] = 0

markers_copy = markers.copy()
markers_copy[markers==0] = 150  # 灰色表示背景
markers_copy[markers==1] = 0    # 黑色表示背景
markers_copy[markers>1] = 255   # 白色表示前景

markers_copy = np.uint8(markers_copy)

cv2.imshow('markers_copy',markers_copy)
markers = cv2.watershed(img, markers)
img[markers==-1] = [0,0,255]
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()