# Project 5 - Extended Kalman Filter

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)  

Overview
---

In this project, I utilized an `Extended Kalman Filter` to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower than the tolerance outlined in the project rubric.

<p align="center">
<img width="500" height="350" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%205%20-%20Extended%20Kalman%20Filter/kalman_filter_map.png"
</p>

Background - Radar
---

* Radar has been used in automotive for years, usually placed behind the bumpers on light-duty vehicles.  
* Radar utilizes the Doppler effect to measure speed (velocity). 
* Can be used for localization, generating radar maps of the environment.  Can provide distance to objects without a direct line of sight.  Can also see under vehicles & buildings, & other objects that could otherwise be obsecured.
* Least affected by rain or fog, such as Lidar or Cameras  
* Wide filed of view ~ 150 degrees
* Range is 200 + metres  
* Resultion in vertical dimension is limited  
* Debris such as pop cans can have high clutter, high radar reflection, because of this auto manufacturers avoid static objects.



Dependencies
---

This project requires [C++](https://isocpp.org/) and the following C++ libraries installed:  
  
  Eigen Library - [git clone](https://gitlab.com/libeigen/eigen.git) 
  
File Structure
---

`main.cpp` - communicates with the Term 2 Simulator receiving data measurements, calls a function to run the Kalman filter, calls a function to calculate RMSE  
`FusionEKF.cpp` - initializes the filter, calls the predict function, calls the update function  
`kalman_filter.cpp`- defines the predict function, the update function for lidar, and the update function for radar  
`tools.cpp`- function to calculate RMSE and the Jacobian matrix  
 
Algorithim
---

The Kalman Filter algorithm will go through the following steps:  

**first measurement** - the filter will receive initial measurements of the bicycle's position relative to the car. These measurements will come from a radar or lidar sensor.  


**initialize state and covariance matrices** - the filter will initialize the bicycle's position based on the first measurement.  

then the car will receive another sensor measurement after a time period Δt. 

**predict** - the algorithm will predict where the bicycle will be after time Δt. One basic way to predict the bicycle location after \Delta{t}Δt is to assume the bicycle's velocity is constant; thus the bicycle will have moved velocity Δt.  We will assume the velocity is constant.  

**update** - the filter compares the "predicted" location with what the sensor measurement says. The predicted location and the measured location are combined to give an updated location. The Kalman filter will put more weight on either the predicted location or the measured location depending on the uncertainty of each value.  

then the car will receive another sensor measurement after a time period Δt. The algorithm then does another predict and update step.
 
  
Remarks
---

