import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import skimage as ski

def Waterhed_pill():
    root=os.getcwd()
    imagePath=os.path.join('testImages/coin.png')
    img=cv2.imread(imagePath)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(331)
    plt.imshow(img,cmap='gray')

    _,imgThred=cv2.threshold(img,155,255,cv2.THRESH_BINARY)#cv2.THRESH_OTSU 会自动选择一个最优的阈值。
    imgThred=255-imgThred
    plt.subplot(332)
    plt.imshow(imgThred,cmap='gray')


    backgroud=ski.morphology.erosion(imgThred, ski.morphology.square(7)).astype(np.double)
    maske=imgThred.astype(np.double)
    recon=ski.morphology.reconstruction(backgroud, maske, 'dilation').astype(np.double)
    plt.subplot(333)
    plt.imshow(recon,cmap='gray')

    recon=np.uint8(recon)
    holeless = ski.morphology.remove_small_holes(recon, area_threshold=5)
    plt.subplot(335)
    plt.imshow(holeless,cmap='gray')
    recon[holeless == 1] = 255
    plt.subplot(334)
    plt.imshow(recon,cmap='gray')



    imgDist=cv2.distanceTransform(recon,cv2.DIST_L2,5)

    local_max = ski.feature.peak_local_max(imgDist, min_distance=10)
    seed = np.zeros_like(recon)
    seed[tuple(local_max.T)] = 255 #.T表示转置的意思

    # cv2.normalize(imgDist, imgDist, 0, 1.0, cv2.NORM_MINMAX)
    # ret, sure_fg2 = cv2.threshold(imgDist, 0.5 * imgDist.max(), 255, 0)
    # sure_fg2 = np.uint8(sure_fg2)
    # plt.subplot(338)
    # plt.imshow(sure_fg2)

    plt.subplot(336)
    plt.imshow(seed,cmap='gray')
    flood_area = cv2.subtract(recon, seed)

    seed=np.uint8(seed)
    _, labels = cv2.connectedComponents(seed)
    labels = labels + 1
    labels[flood_area == 255] = 0
    labels = np.int32(labels)

    plt.subplot(337)
    plt.imshow(flood_area,cmap='gray')

    recon=cv2.cvtColor(recon,cv2.COLOR_BGR2RGB)
    labels = cv2.watershed(recon, labels)#关键点！！！！！！永远原图去分水岭的话会出现中间空的现象
    imgRGB[labels == -1] = [255, 0, 0]
    plt.subplot(339)
    plt.imshow(imgRGB)




    plt.show()


if __name__=='__main__':
    Waterhed_pill()