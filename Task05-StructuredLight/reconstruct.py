# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys


def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")


def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0), fx=scale_factor,
                           fy=scale_factor)
    ref_clr = cv2.resize(cv2.imread("images/pattern001.jpg", 1) , (0, 0), fx=scale_factor,
                         fy=scale_factor)
#    ref_clr = cv2.cvtColor(ref_clr,cv2.COLOR_BGR2RGB)


    ref_avg = (ref_white + ref_black) / 2.0
    ref_on = ref_avg + 0.05  # a threshold for ON pixels
    ref_off = ref_avg - 0.05  # add a small buffer region

    h, w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h, w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0, 15):
        # read the file
        patt = cv2.resize(cv2.imread("images/pattern%03d.jpg" % (i + 2), cv2.IMREAD_GRAYSCALE) /255.0, (0, 0), fx=scale_factor, fy=scale_factor)

        # mask where the pixels are ON
        on_mask = (patt > ref_on) & proj_mask
        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)

        # TODO: populate scan_bits by putting the bit_code according to on_mask
        for i in range(len(proj_mask)):
            for j in range(len(proj_mask[i])):
                if proj_mask[i][j]:
                    if on_mask[i][j]:
                        scan_bits[i][j] += bit_code

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl", "r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    camera_points = []
    projector_points = []
    color = []

    img = np.zeros((h, w, 3), 'uint8')
    for x in range(w):
        for y in range(h):
            if not proj_mask[y, x]:
                continue  # no projection here
            if scan_bits[y, x] not in binary_codes_ids_codebook:
                continue  # bad binary code

            p_x, p_y = binary_codes_ids_codebook[scan_bits[y,x]]

            # ... obtain x_p, y_p - the projector coordinates from the codebook
            if p_x >= 1279 or p_y >= 799:  # filter
                continue

            img[y, x] = (0, p_y*255/h, p_x*255/w)
            camera_points.append([[x/2, y/2]])
            projector_points.append([[p_x,p_y]])
            color.append([ref_clr[y, x]])
            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points

    cv2.imwrite("correspondence.jpg",img)

    # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
#    print(projector_points)
#    print(camera_points)
    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl", "r") as f:
        d = pickle.load(f)
        camera_K = d['camera_K']
        camera_d = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']


        # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
        cam = cv2.undistortPoints(np.array(camera_points,dtype=np.float32), np.array(camera_K,dtype=np.float32),np.array(camera_d,dtype=np.float32))

        # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
        proj = cv2.undistortPoints(np.array(projector_points,dtype=np.float32), np.array(projector_K,dtype=np.float32), np.array(projector_d,dtype=np.float32))

        # TODO: use cv2.triangulatePoints to triangulate the normalized points
#        pts = cv2.triangulatePoints(np.ones((3,4)),np.concatenate((projector_R, projector_t), axis=1),cam,proj)
        #np.eye()

        pts = cv2.triangulatePoints(np.concatenate((np.identity(3), np.zeros((3,1))), axis=1),np.concatenate((projector_R, projector_t), axis=1),cam,proj)

        # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
        points_3d_o = cv2.convertPointsFromHomogeneous(pts.T)
        #print(points_3d)

        # TODO: name the resulted 3D points as "points_3d"

        points_3d = np.array([points_3d_o[np.where((points_3d_o[:, :, 2] > 200) & (points_3d_o[:, :, 2] < 1400))]])
        points_3d = np.swapaxes(points_3d, 0, 1)
        #print(points_3d)
        #print(color)
        color = np.array(color)
        color = np.array([color[np.where((points_3d_o[:, :, 2] > 200) & (points_3d_o[:, :, 2] < 1400))]])
        color = np.swapaxes(color, 0, 1)
        concat = np.concatenate((points_3d,color),2)
        with open(sys.argv[1] +"output_rgb.xyz", "w") as f:
            for p in concat:
                f.write("%d %d %d %d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2], p[0,5], p[0,4], p[0,3]))

        return points_3d



def write_3d_points(points_3d):
    # ===== DO NOT CHANGE THIS FUNCTION =====
    print("write output point cloud")


    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name, "w") as f:
        for p in points_3d:
            f.write("%d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2]))

    return points_3d

if __name__ == '__main__':

    # ===== DO NOT CHANGE THIS FUNCTION =====

    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)