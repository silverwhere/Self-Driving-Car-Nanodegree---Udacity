# Project 1: Finding Lane Lines on the Road

---
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### **Finding Lane Lines on the Road**

Conventional vehicles from the very first steam engines to this centuries internal combustion engines and electric vehicles have relied on human decision making by the driver of the vehicle to identify and navigate the lanes of the road.  As automonous vehicles will have to be able to precieve, decide and action to target their position automatically.  To accomplish this we first will need to develop a pipeline using computer vision methods utlized from the OpenCV libary for Python.  The libary Open-Source Computer Vision (OpenCV) is a library of programming functions mainly aimed at real-time computer vision.  It is the combination of this library and how to best process the video images we will be able to identify the location of the lanes in real-time.


[//]: # (Image References)

![Figure1](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%201%20-%20Finding%20Lane%20Lines/test_images/solidYellowCurve.jpg)

---

### 1.0 Description

The project consists of first developing a pipeline for use on several still images of 960 x 540 pixels to train our pipeline against. Below is one of the provided images.

![Figure2](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%201%20-%20Finding%20Lane%20Lines/test_images/solidWhiteCurve.jpg)

After the pipeline is proven to work on still images of our video feed, we can apply the pipeline to full video to determine if the identification of lane lines is sufficient.

#### Step 1: Convert RGB (Red, Green, Blue) Image to HSL (Hue, Saturation, Lightness)

The provided images are in RGB colorspace. To begin processing these images, I first considered converting the images to grayscale.  Grayscale images remove the color and focus on the areas of highest contrast, such as a white or yellow line when against the backdrop of a black asphalt road, i.e. the white and yellow lines will be brighter than the road and therefore observe a higher contrast.  Recall however though that if I were to filter the image out to individual color channels "Red, Green and Blue", the yellow lane lines would not appear in the "Green" color channel and therefore would be a good technique if I was to program in what each color lane line would represent, though seperating yellow and white lanes not required for this project.   

Continuing, I achieved a decent result with RGB to Grayscale conversion, but I had also learned about HSV and HSL colorspace when developing a traffic light classifier In the "Intro to Self Driving Cars Nanodegree" to detect specific colors of a traffic light.  At this point I am only concerned with detecting a line, not the specific color, so HSV (Hue, Saturation, Value) or HSL (Hue, Saturation, Lightness) colorspace will prove to be the best filter to detect the lines themselves.  Utilzing HSL I am better able to detect the lines as "lightness" is best understood as the amount of white in the pixel.  HSV provided a better result than grayscale, however, when looking at the results of HSV compared to HSL, I found better detection and proceeded to continue with HSL for accuracy.

![Figure3](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%201%20-%20Finding%20Lane%20Lines/test_pipeline_images/gray_white_lanes.jpg)*Grayscale*  
![Figure4](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%201%20-%20Finding%20Lane%20Lines/test_pipeline_images/hsv_white_lanes.jpg)*HSV*  
![Figure5](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%201%20-%20Finding%20Lane%20Lines/test_pipeline_images/hls_white_lanes.jpg)*HLS*  

#### Step 2: APPLY A GAUSSIAN BLUR FILTER FOR SMOOTHING OF LANE LINES

In image processing, a Gaussian blur (also known as Gaussian smoothing) is an image pre-processing technique.  It is the result of blurring an image by a Gaussian function (named after mathematician and scientist Carl Friedrich Gauss).  The effect is typically to reduce image noise and reduce detail.

![Figure6](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%201%20-%20Finding%20Lane%20Lines/test_pipeline_images/gaussian_blur.jpg)

*Note that a Canny Filter which we will also use has a 5 x 5 Gaussian Blur, but adding one before is for additional smoothing.

#### STEP 3: APPLY A CANNY EDGE DETECTOR

A Canny Edge Detector is an edge detection operator.  This is useful for us as since we have already identified the regions of lightness and smoothed the image in pre-processing, the Canny Edge Detector can detect and edge with a low error rate, which means that the detection should accurately catch as many edges shown in the image as possible.  The edge point detected from the operator should accurately localize on the center of the edge.  Finally a given edge in the image should only be marked once, and where possible, image noise should not create false edges.

![Figure7](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%201%20-%20Finding%20Lane%20Lines/test_pipeline_images/canny_edge.jpg)

#### STEP 4: CREATE A MASKED IMAGE OF OUR CANNY EDGE OUTPUT

The output of the Canny Edge Detector is an image "edges", this of course has edges detected in the entire image, which includes areas not of interest such as other lane lines, or road signs and trees.  To focus in directly on the problem, I applied a region of interest utilizing a polygon combined with the masked image.  The result is a Canny Edge Output image "masked_edges".  

![Figure8](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%201%20-%20Finding%20Lane%20Lines/test_pipeline_images/region_of_interest.jpg)

The OpenCV implementation requires passing in two parameters in addition to our blurred image, a low and high threshold which determines whether to include a given edge or not. A threshold captures the intensity of change of a given point (you can think of it as a gradient). Any point beyond the high threshold will be included in our resulting image, while points between the threshold values will only be included if they are next to edges beyond our high threshold. Edges that are below our low threshold are discarded. Recommended low:high threshold ratios are 1:3 or 1:2. We use values 50 and 150 respectively for low and high thresholds.

![Figure9](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%201%20-%20Finding%20Lane%20Lines/test_pipeline_images/masked_canny.jpg)

#### STEP 5: PERFORM A HOUGH TRANSFORM 

The next step is to apply the Hough Transform technique to extract lines and color them. A Hough Transform finds lines by identifying all points that lie on them. This is done by converting our current Cartesian system denoted by axis (x,y) to a parametric one where axes are (m (slope), b (y-intercept). 

In parameter space this plane:
1. lines are represented as points
2. points are presented as lines (since they can be on many lines in traditional coordinate system)
3. intersecting lines means the same point is on multiple lines

Therefore, in such plane, we can more easily identify lines that go via the same point. We however need to move from the current system to a Hough Space which uses a polar coordinates system as our original expression is not differentiable when m=0 (i.e. vertical lines). In polar coordinates, a given line will now be expressed as (ρ, θ), where line L is reachable by going a distance ρ at angle θ from the origin, thus meeting the perpendicular L; that is ρ = x cos θ + y sin θ.  

All straight lines going through a given point will correspond to a sinusoidal curve in the (ρ, θ) plane. Therefore, a set of points on the same straight line in Cartesian space will yield sinusoids that cross at the point (ρ, θ). This naturally means that the problem of detecting points on a line in cartesian space is reduced to finding intersecting sinusoids in Hough space.

![Figure10](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%201%20-%20Finding%20Lane%20Lines/test_pipeline_images/hough_lines.jpg)

#### STEP 6: SEPERATE LEFT AND RIGHT LANES / DRAW LINES

The drawlines function that was provided works quite well, but it only draws lines over the edges detected and often you end up with a line on each edge of the solid road line and not a single solid line.  To solve for this I noticed that while the lanes are in fact straight on the road, our camera viewing angle actually shows a slope for both the left lane (positive slope) and right lane (negative slope).  In order to draw a single line on the left and right lanes, I modified the draw_lines() function by first calculating the slope of the detected lines coordinates.  For slopes greater or less than 0.3 I appended the coordinates of the identified left or right lane lines.  Slopes between -0.3 and 0.3 were ignored as edges close to zero did not appear as a lane line and often detected edges that were not lanes in certain frames of the video.

Now that we know our left and right lanes, we can draw single, solid, red lines that trace the lane line through the entire region of interest.  To accomplish this we need to determine the X-coordinates of the bottom of the line and the top of the line to be traced.  Y-coordinates were already determined as the y-coodrinates of the region of interest, 540 (bottom) and 350 (top) pixels.  To draw the lines we used our x and y coordinates with the CV2.line function to draw our solid red lines in red.

![Figure11](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%201%20-%20Finding%20Lane%20Lines/test_pipeline_images/solidWhiteCurve.jpg) 
### 2.0 Identify potential shortcomings with your current pipeline

I feel my pipeline accurately determines the location of the left or right lane accurately, this includes a lane that is painted yellow or white.  A shortcoming to my pipeline is the accurate determination of curved lines.  My pipeline is detecting the length of a line first before determining how long of a solid red line is should draw.  


### 3.0 Suggest possible improvements to your pipeline

To improve my pipeline I could seperate my original image into RGB color channels.  Yellow would only appear in a green or red channel and not a blue channel.  Knowing this I could isolate my white lines from my yellow lines and potentially apply logic and rules for solid and dashed yellow or white lines, or orange lines is often new in areas under construction!

With regards to curved lines following a curve in the road, thinking about this, my pipeline is focusing on my area of interest, which my polygon might be too large.  Instead, I might want several polygons, slicing up a portion of the road horizontally.  To detect the curve, I might also borrow a strategy from a previous project on a traffic light classifier, utlizing a histogram filter to detect changes in brightness combined with a Canny Edge Detector.  


