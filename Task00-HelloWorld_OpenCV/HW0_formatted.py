import cv2
import numpy as np


def readImgBW(path):
    return cv2.imread(path, 0)


def readImgColor(path):
    return cv2.imread(path, 1)


def displayImg(title, img):
    cv2.imshow(title, img.astype('uint8'))
    cv2.waitKey()
    cv2.destroyAllWindows()


def resizeByFactor(img, factor):
    height, width, dim = img.shape
    return cv2.resize(
        img, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_AREA)


img = readImgColor("CVMeme.jpg")

#display original image
displayImg("Original image... " + str(img), img)

#adding 30 to each pixel
addImg = img + 30
displayImg("added 30 to each pixel... " + str(addImg), addImg)

#substracting 30 from each pixel
subImg = img - 30
displayImg("substracted 30 from each pixel... " + str(subImg), subImg)

#multiplied by 2
mulImg = img * 2
displayImg("multiplied by 2 to each pixel... " + str(mulImg), mulImg)

#diving by 2
divImg = img / 2
displayImg("divided by 2 to each pixel... " + str(divImg), divImg)

#resized image
resizedImg = resizeByFactor(img, 2)
displayImg("resized image uniformly by 1/2 ", resizedImg)
