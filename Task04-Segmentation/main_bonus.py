import cv2
import numpy as np
import sys
# ================================================
# Skeleton codes for HW4
# Read the skeleton codes carefully and put all your
# codes into main function
# ================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay
import sys

drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve

# mouse callback function
def draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode, colour, prevColor, fg, bg
    if event == cv2.EVENT_RBUTTONUP:

        temp = colour
        colour = prevColor
        prevColor = temp

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),colour,5)
                if colour == (0, 0, 255):
                    bg.append((current_former_x,current_former_y))
                else:
                    fg.append((current_former_x, current_former_y))

                current_former_x = former_x
                current_former_y = former_y
                #print former_x,former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),colour,5)
            if colour == (0, 0, 255):
                bg.append((current_former_x, current_former_y))
            else:
                fg.append((current_former_x, current_former_y))

            current_former_x = former_x
            current_former_y = former_y


    return former_x,former_y




def help_message():
   print("Usage: [Input_Image] [Input_Marking] [Output_Directory]")
   print("[Input_Image]")
   print("Path to the input image")
   print("[Input_Marking]")
   print("Path to the input marking")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " astronaut.png " + "astronaut_marking.png " + "./")

# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=22)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:,0]!=255])
    bg_segments = np.unique(superpixels[marking[:,:,2]!=255])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                                  cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)


def launchWindow():


    cv2.namedWindow("OriginalImage")
    cv2.setMouseCallback('OriginalImage', draw)
    cv2.imshow('OriginalImage', im)

    while (1):
        cv2.imshow('OriginalImage', im)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 :
            break
        if key == 32:
            mask = np.copy(im)
            mask[:] = 255
            for (j, i) in fg:
                for k in (-3, -2, -1, 0, 1, 2, 3):
                    if ((j + k) in range(0, 256)) and ((i + k) in range(0, 256)):
                        mask[i, j][2] = 255
                        mask[i, j][0] = 0
                        mask[i, j][1] = 0

            for (j, i) in bg:
                for k in (-3, -2, -1, 0, 1, 2, 3):
                    if ((j + k) in range(0, 256)) and ((i + k) in range(0, 256)):
                        mask[i, j][2] = 0
                        mask[i, j][0] = 255
                        mask[i, j][1] = 0

            segmented = segment(mask)


            cv2.imshow("SegmentedImage", segmented)
            cv2.waitKey(0)

            cv2.destroyWindow("SegmentedImage")

    cv2.destroyAllWindows()

def segment(mask):
    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(img)
    fg_segments, bg_segments = find_superpixels_under_marking(mask, superpixels)
    fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)
    bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, color_hists)
    fgbg_hists = [fg_cumulative_hist, bg_cumulative_hist]
    fgbg_superpixels = [fg_segments, bg_segments]

    norm_hists = normalize_histograms(color_hists)

    graph_cut = do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors)
    graph_cut = graph_cut.astype('int')

    segmask = pixels_for_segment_selection(superpixels, np.nonzero(graph_cut))
    segmask = np.uint8(segmask * 255)

    return segmask



fg = []
bg = []
im = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
img = im[:]



prevColor = (0, 0, 255)
colour = (255, 0, 0)

print("Press ESC key to exit")
print("BLUE - foreground, RED - background")
print("To change color right click once")
print("To see result, press space bar")
launchWindow()
