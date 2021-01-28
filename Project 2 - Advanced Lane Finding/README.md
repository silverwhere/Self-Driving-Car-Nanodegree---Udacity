## Advanced Lane Finding
# Project 2: Advanced Lane Finding

---
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### **Advanced Lane Finding**

![alt text](videos/project_video_augmented.gif "Result")

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

![Figure1](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%202%20-%20Advanced%20Lane%20Finding/output_images/calibration17.jpg)

The project consists of first developing a pipeline

### 1.0 Camera Calibration

Camera parameters include intrinsics, extrinsics (camera matrix 'mtx'), and distortion coefficients ('dist'). To estimate the camera parameters, you need to have 3-D world points and their corresponding 2-D image points. You can get these correspondences using multiple images of a calibration pattern, such as a chessboard. Using the correspondences, you can solve for the camera parameters.

Camera parameters include intrinsics, extrinsics, and distortion coefficients. To estimate the camera parameters, you need to have 3-D world points and their corresponding 2-D image points. You can get these correspondences using multiple images of a calibration pattern, such as a checkerboard. Using the correspondences, you can solve for the camera parameters. 

I developed a camera calibration function when Camera calibration, given object points, image points, and the shape of the grayscale image, returns distortion coefficients (dist) camera matrix (mtx), camera position in the world, rotation (rvecs), translation (tvecs).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:



### 2.0 Image Distortion Correction

The lens used in a camera is not a perfect piece of glass, so some form of distortion might be present in the image it captures. There is what we call radial distortion in which the light bends at a certain angle that deviates from a rectilinear plane.  Therefore, we need to correct for image distortion utilizing our camera matrix and distortion coefficients.  Distortion can - 

* change the apparent size of an object in an image.  
* change the apparent shape of an object in an image.  
* cause an objects appearance to change depending on where it is in the field of view.  
* and make objects appear closer or father away than they actually are.

I utilized `cv2.undistort(img, mtx, dist, None, mtx)` to perform an image distortion correction.
       

