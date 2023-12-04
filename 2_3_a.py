import numpy as np
import math
import scipy
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time


# ---------------------------------------------------------------------------------------------------
# main
def watershed_full():
    img_size = (300, 300)
    img = np.zeros(img_size, dtype=np.uint8)

    # Define the parameters for three circles
    circle1 = (100, 100, 50)
    circle2 = (180, 120, 40)
    circle3 = (60, 200, 35)

    # Draw three circles on the image
    cv2.circle(img, (circle1[0], circle1[1]), circle1[2], 255, -1)
    cv2.circle(img, (circle2[0], circle2[1]), circle2[2], 255, -1)
    cv2.circle(img, (circle3[0], circle3[1]), circle3[2], 255, -1)

    dmap = cv2.distanceTransform(img, cv2.DIST_L2, 3)  # Distance map wird erstellt
    local_max = ski.feature.peak_local_max(dmap, min_distance=10)  # Locale Maximas aus Distance map

    plt.figure()
    plt.subplot(231)
    plt.imshow(dmap,cmap='gray')

    plt.subplot(232)
    plt.imshow(local_max)

    seed = np.zeros_like(img)
    seed[tuple(local_max.T)] = 255  # Locale Maximas in Seed-Bild einfügen
    flood_area = cv2.subtract(img,seed)  # Flood area entspricht der Zone zwischen der Maximas und dem schwarzen (=0) Hintergrund
    # hier wird geflutet

    plt.subplot(233)
    plt.imshow(seed)

    plt.subplot(234)
    plt.imshow(flood_area,cmap='gray')

    _, labels = cv2.connectedComponents(seed)  # Alles Maximas nummerieren
    labels = labels + 1  # Labels werden um +1 erhöht, sodass die Label-Nummer 0 frei wird

    labels[flood_area == 255] = 0  # Alles was jetzt im Labelbild 0 ist wird geflutet

    labels = np.int32(labels)
    #img = cv2.merge((img, np.zeros_like(img),np.zeros_like(img)))  # OpenCV watershed ist doof und braucht zwingend ein 3-channel Bild
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    watershed = cv2.watershed(imgRGB, labels)
    imgRGB[labels==-1] = [255,0,0]
    plt.subplot(235)
    plt.imshow(watershed)

    plt.subplot(236)
    plt.imshow(imgRGB)


    #seg_img = np.uint8(seg_img)
    # if switch == 0:
    #     return seg_img
    # elif switch == 1:
    #     return watershed.astype(np.int32)

    plt.show()

if __name__=='__main__':
   watershed_full()
# ---------------------------------------------------------------------------------------------------
# output

# segmented_img_uint8 = watershed_full(image, 0)
# segmented_img_int32 = watershed_full(image, 1)
#
# cv2.imshow('Watershed', segmented_img_uint8)
# cv2.imshow('Orginal', image)

# plot_image_to_3D(segmented_img_int32)

# plt.figure()
# plt.imshow(segmented_img_int32)
# plt.show()

# plot_image_to_3D(labels)


# ---------------------------------------------------------------------------------------------------
# main-end

# plt.show()
# print('La fin')
# cv2.waitKey(0)