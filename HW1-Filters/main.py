# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy


def help_message():
    print("Usage: [Question_Number] [Inumpyut_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Inumpyut_Options]")
    print("Path to the inumpyut images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to inumpyut image] " + "[output directory]")  # Single inumpyut, single output
    print(sys.argv[
              0] + " 2 " + "[path to inumpyut image1] " + "[path to inumpyut image2] " + "[output directory]")  # Two inumpyuts, three outputs
    print(sys.argv[
              0] + " 3 " + "[path to inumpyut image1] " + "[path to inumpyut image2] " + "[output directory]")  # Two inumpyuts, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================

def histogram_equalization(img_in):
    channels = cv2.split(img_in)
    newChannels = []
    for channel in channels:
        hist, bins = numpy.histogram(channel, 256)
        cdf = hist.cumsum() * 1.0
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
        newChannels.append(cdf[channel])

    img_out = cv2.merge((newChannels[0], newChannels[1], newChannels[2]))

    return True, img_out


def Question1():
    # Read in inumpyut images
    inumpyut_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);

    # Histogram equalization
    succeed, output_image = histogram_equalization(inumpyut_image)

    # Write out the result
    output_name = sys.argv[3] + "1.jpg"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================

def low_pass_filter(img_in):
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # fft and shift it to the center
    f = numpy.fft.fft2(img_in)
    fshift = numpy.fft.fftshift(f)

    # apply filter and inverse filter
    r, c = img_in.shape
    fshift[0:int(r / 2) - 10, 0:int(c / 2) + 10] = 0
    fshift[int(r / 2) + 10: r, 0:int(c / 2) + 10: c] = 0
    fshift[int(r / 2) - 10: int(r / 2) + 10, 0:int(c / 2) - 10] = 0
    fshift[int(r / 2) - 10: r, int(c / 2) + 10:c] = 0
    f_ishift = numpy.fft.ifftshift(fshift)
    img_out = numpy.fft.ifft2(f_ishift)
    img_out = numpy.abs(img_out)

    return True, img_out


def high_pass_filter(img_in):
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # fft and shift it to the center
    f = numpy.fft.fft2(img_in)
    fshift = numpy.fft.fftshift(f)

    # apply filter and inverse filter
    r, c = img_in.shape
    fshift[int(r / 2) - 10:int(r / 2) + 10, int(c / 2) - 10:int(c / 2) + 10] = 0
    f_ishift = numpy.fft.ifftshift(fshift)
    img_out = numpy.fft.ifft2(f_ishift)
    img_out = numpy.abs(img_out)

    return True, img_out


def ft(im, newsize=None):
    dft = numpy.fft.fft2(numpy.float32(im), newsize)
    return numpy.fft.fftshift(dft)


def ift(shift):
    f_ishift = numpy.fft.ifftshift(shift)
    img_back = numpy.fft.ifft2(f_ishift)
    return numpy.abs(img_back)


def deconvolution(img_in):
    # Write deconvolution codes here
    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T

    imf = ft(img_in, (img_in.shape[0], img_in.shape[1]))  # make sure sizes match
    gkf = ft(gk, (img_in.shape[0], img_in.shape[1]))  # so we can multiple easily
    imconvf = imf / gkf * 255

    # now for example we can reconstruct the blurred image from its FT
    img_out = ift(imconvf)

    return True, img_out


def Question2():
    # Read in inumpyut images
    inumpyut_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
    #   inumpyut_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
    inumpyut_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(inumpyut_image1)
    succeed2, output_image2 = high_pass_filter(inumpyut_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(inumpyut_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2.jpg"
    output_name2 = sys.argv[4] + "3.jpg"
    output_name3 = sys.argv[4] + "4.jpg"
    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):
    G = img_in1.copy()
    gpA = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpA.append(G)
    # generate Gaussian pyramid for B
    G = img_in2.copy()
    gpB = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpB.append(G)
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]

    for i in xrange(5, 0, -1):
        size = (gpA[i].shape[1], gpA[i].shape[0])
        GE = cv2.pyrUp(gpA[i], dstsize=size)
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in xrange(5, 0, -1):
        size = (gpB[i].shape[1], gpB[i/].shape[0])
        GE = cv2.pyrUp(gpB[i], dstsize=size)
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)
    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = numpy.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1, 6):
	size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = cv2.add(ls_, LS[i])
    # image with direct connecting each half
    real = numpy.hstack((img_in1[:, :cols / 2], img_in2[:, cols / 2:]))

    img_out = ls_
    return True, img_out


def Question3():
    # Read in inumpyut images
    inumpyut_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
    inumpyut_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(inumpyut_image1, inumpyut_image2)

    # Write out the result
    output_name = sys.argv[4] + "5.jpg"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the inumpyut arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])
        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Inumpyut parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
