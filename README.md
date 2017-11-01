# Playing with OpenCV

### Task00:HelloWorld_OpenCV
  - Reading/Writing image using openCV
  - manipulate each pixel to add/sub/mul/div each pixel

### Task01-Filters_HistEqv
  - HighPass Filter
  - LowPass Filter
  - Histogram equalizer
	-  Deconvolution
  - Image Blending

### Task02-ImageWarping
  Image Alignment, Panoramas
  homographies and perspective warping on a common plane (3 images).
  cylindrical warping (many images).

### Task03-ObjectDetectionFromVideo
  - Detect the face in the first frame of the movie Using pre-trained Viola-Jones detector
  - Track the face throughout the movie using:
    - CAMShift
    - Particle Filter
    - Face detector + Kalman Filter
    - Optical Flow tracker
 
### Task4-Segmentation
  - perform semi-automatic binary segmentation based on SLIC superpixels and graph-cuts
   - Given an image and sparse markings for foreground and background
   - Calculate SLIC over image
   - Calculate color histograms for all superpixels
   - Calculate color histograms for FG and BG.
   - Construct a graph that takes into account superpixel-to-superpixel interaction (smoothness term), as well as superpixel-FG/BG interaction (match term)
   - Run a graph-cut algorithm to get the final segmentation

  - Make it interactive: Let the user draw the markings (carrying 0 pt for this part)
   - for every interaction step (mouse click, drag, etc.)
   - recalculate only the FG-BG histograms,
   - construct the graph and get a segmentation from the max-flow graph-cut, 
   - show the result immediately to the user (should be fast enough).
