
# Project 2: Advanced Lane Finding

---
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### **Advanced Lane Finding**

![alt text](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%202%20-%20Advanced%20Lane%20Finding/project_video_out.gif.gif "Result")

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)
---  
The project consists of first developing a pipeline

### 1.0 Camera Calibration

Camera parameters include intrinsics, extrinsics (camera matrix 'mtx'), and distortion coefficients ('dist'). To estimate the camera parameters, you need to have 3-D world points and their corresponding 2-D image points. You can get these correspondences using multiple images of a calibration pattern, such as a chessboard. Using the correspondences, you can solve for the camera parameters.

Camera parameters include intrinsics, extrinsics, and distortion coefficients. To estimate the camera parameters, you need to have 3-D world points and their corresponding 2-D image points. You can get these correspondences using multiple images of a calibration pattern, such as a checkerboard. Using the correspondences, you can solve for the camera parameters. 

I developed a camera calibration function when Camera calibration, given object points, image points, and the shape of the grayscale image, returns distortion coefficients (dist) camera matrix (mtx), camera position in the world, rotation (rvecs), translation (tvecs).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Figure1](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%202%20-%20Advanced%20Lane%20Finding/output_images/calibration17.jpg)


### 2.0 Image Distortion Correction

The lens used in a camera is not a perfect piece of glass, so some form of distortion might be present in the image it captures. There is what we call radial distortion in which the light bends at a certain angle that deviates from a rectilinear plane.  Therefore, we need to correct for image distortion utilizing our camera matrix and distortion coefficients.  Distortion can - 

* change the apparent size of an object in an image.  
* change the apparent shape of an object in an image.  
* cause an objects appearance to change depending on where it is in the field of view.  
* and make objects appear closer or father away than they actually are.

I utilized `cv2.undistort(img, mtx, dist, None, mtx)` to perform an image distortion correction.

![Figure2](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%202%20-%20Advanced%20Lane%20Finding/output_images/distorted_undistored.jpg)

### 3.0 Apply a Perspective Transform

Transforming an image such that we are effectively viewing objects from a different angle or direction.  Objects appear smaller the farther away they are from a viewpoint, like a camera, parallel lanes appear to converge to a point.  We are interested in a perspective transform as we want to ultimately measure the curvature of a lane line.
To accomplish this I needed to transform a top-down view using `cv2.getPerspectiveTransform()` to obtain the to get `M`, the transform matrix and `Minv` the inverse transform matrix, to be utilized later to transform our image back.  Prior to transforming my image I had to select source points `src` and destination points `dst`.   In Perspective Transform, we need provide the points `src` on the image from which want to gather information by changing the perspective. We also need to provide the points inside which we want to display our image `dst`. Then, we get the perspective transform from the two given set of points and wrap it with the original image using ` cv2.warpPerspective()`  to warp your image to a top-down view.

![Figure3](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%202%20-%20Advanced%20Lane%20Finding/output_images/Undistored_Warped.jpg)

### 4.0 Thresholding

In digital image processing, thresholding is the simplest method of segmenting images. From a grayscale image, thresholding can be used to create binary images. 

### 4.1 Sobel Operator Threshold

Sobel operators is a joint Gausssian smoothing plus differentiation operation, so it is more resistant to noise. You can specify the direction of derivatives to be taken, vertical y or horizontal x.  The Sobel operator is at the heart of the Canny edge detection algorithim.  Applying the Sobel operator to an image is
a way of taking the derivative of the image in the x or y direction.  As lane lines are mostly vertical, I choose to use a Sobelx, denoted by 1,0, `cv2.Sobel(gray, cv2.CV_64F, 1, 0)` whereas Sobely would be 0,1.  I converted the image to 8-bit and a binary threshold to select pixels based on gradient strength from 20 to 100.  The result was lane lines identified as follows:

![Figure4](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%202%20-%20Advanced%20Lane%20Finding/output_images/Undistored_Warped.jpg)

### 4.2 Magnitude of the Gradient

Apply a threshold to the overall magnitude of the gradient, in both x and y direction by taking the square root of Sobelx^2 and Sobely^2 `np.sqrt(sobelx**2 + sobely**2)`

![Figure5](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%202%20-%20Advanced%20Lane%20Finding/output_images/Mag_Gradient.jpg)
