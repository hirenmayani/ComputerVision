import cv2
import numpy as np
from matplotlib import pyplot as plt


def displayImg(title, img):
    cv2.imshow(title,img.astype('uint8'))
    cv2.waitKey()
    cv2.destroyAllWindows()

img = cv2.imread('input2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

rows, cols = img.shape
crow,ccol = int(rows/2) , int(cols/2)
fshift[crow-10:crow+10, ccol-10:ccol+10] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

displayImg("op", img_back)