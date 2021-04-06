# Project 5 - Extended Kalman Filter

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)  

Overview
---

In this project, I utilized end-to-end deep learning using convolutional neural networks (CNNs) to map the raw pixels from (3) front-facing cameras to the steering commands for a self-driving car. A simulator was used to capture the images during training laps around a track with various turns, curb styles, heights and pavement. This powerful end-to-end approach means that with minimum training data from humans, the system learns to steer, with or without lane markings on roads.

<p align="center">
<img width="500" height="250" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%205%20-%20Extended%20Kalman%20Filter/kalman_filter_map.png"
</p>



Dependencies
---

This project requires [C++](https://isocpp.org/) and the following C++ libraries installed:  
  
  Eigen Library - [git clone](https://gitlab.com/libeigen/eigen.git)  
 
Algorithim
---

The Kalman Filter algorithm will go through the following steps:  

**first measurement** - the filter will receive initial measurements of the bicycle's position relative to the car. These measurements will come from a radar or lidar sensor.  


**initialize state and covariance matrices** - the filter will initialize the bicycle's position based on the first measurement.  

then the car will receive another sensor measurement after a time period \Delta{t}Δt. 

**predict** - the algorithm will predict where the bicycle will be after time \Delta{t}Δt. One basic way to predict the bicycle location after \Delta{t}Δt is to assume the bicycle's velocity is constant; thus the bicycle will have moved velocity * \Delta{t}Δt.  We will assume the velocity is constant.  

**update** - the filter compares the "predicted" location with what the sensor measurement says. The predicted location and the measured location are combined to give an updated location. The Kalman filter will put more weight on either the predicted location or the measured location depending on the uncertainty of each value.  

then the car will receive another sensor measurement after a time period \Delta{t}Δt. The algorithm then does another predict and update step.
 
  
Remarks
---

Behavioural Cloning was a great project to truly learn how vehicle systems can learn to drive a vehicle autonomously.  While this project doesn't capture every edge case one could expect in the real world, what it has shown though is how powerful deep-learning is to regressively determine the desired output.  What would have been interesting in this project would have also been to log the vehicle speed.  In reality, you would probably want to go faster on straightaways with minimal steering angle and slower on turns with larger positive or negative steering angles instead of a constant speed throughout.   
