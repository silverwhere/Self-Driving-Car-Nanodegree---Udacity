# Project 5 - Extended Kalman Filter

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)  

Overview
---

In this project, I utilized end-to-end deep learning using convolutional neural networks (CNNs) to map the raw pixels from (3) front-facing cameras to the steering commands for a self-driving car. A simulator was used to capture the images during training laps around a track with various turns, curb styles, heights and pavement. This powerful end-to-end approach means that with minimum training data from humans, the system learns to steer, with or without lane markings on roads.

<p align="center">
<img width="500" height="250" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%204%20-%20Behavioural%20Cloning/behavioral_cloning_gif.gif"
</p>



Dependencies
---

This project requires [C++](https://isocpp.org/) and the following C++ libraries installed:  
  
  Eigen Library - [git clone](https://gitlab.com/libeigen/eigen.git)  
 
  

 
  
Remarks
---

Behavioural Cloning was a great project to truly learn how vehicle systems can learn to drive a vehicle autonomously.  While this project doesn't capture every edge case one could expect in the real world, what it has shown though is how powerful deep-learning is to regressively determine the desired output.  What would have been interesting in this project would have also been to log the vehicle speed.  In reality, you would probably want to go faster on straightaways with minimal steering angle and slower on turns with larger positive or negative steering angles instead of a constant speed throughout.   
