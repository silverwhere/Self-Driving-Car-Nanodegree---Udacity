# Project 5 - Extended Kalman Filter

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)  

Overview
---

In this project, I utilized an `Extended Kalman Filter` and  `Sensor Fusion`  to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower than the tolerance outlined in the project rubric.

<p align="center">
<img width="500" height="350" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%205%20-%20Extended%20Kalman%20Filter/kalman_filter_map.png"
</p>

Background - RADAR - RAdio Detection And Ranging
---

* Radar has been used in automotive for years, usually placed behind the bumpers on light-duty vehicles.  
* Radar utilizes the Doppler effect to measure speed (velocity). 
* Can be used for localization, generating radar maps of the environment.  Can provide distance to objects without a direct line of sight.  Can also see under vehicles & buildings, & other objects that could otherwise be obsecured.
* Least affected by rain or fog, such as Lidar or Cameras  
* Wide filed of view ~ 150 degrees
* Range is 200 + metres  
* Resultion in vertical dimension is limited  
* Debris such as pop cans can have high clutter, high radar reflection, because of this auto manufacturers avoid static objects.  

Background - LIDAR - LIght Detection And Ranging
---  

* LIDAR uses infrared light laser beam to determine the distance between the sensor and nearby object.
* Use light in 900 nano-metre wavelength range (Infrared)  
* Some LIDAR's use longer wavelengths which perform betterin rain or fog.  Can get dirty though and are "bulky!".  
* Data is returned as a point-cloud / high resolution  
* Higher spatial resolution than Radar.   
* Better at vertical scanning versus Radar.


Dependencies
---

This project requires [C++](https://isocpp.org/) and the following C++ libraries installed:  
  
  Eigen Library - [git clone](https://gitlab.com/libeigen/eigen.git) 
  
File Structure
---

* `main.cpp` -  communicates with a simulator that recieves Lidar and Radar data from our vehicle, calls a function to run the kalman filter and calls a function to calculate the error RMSE.  Sends a sensor measurement to `FusionEKF.cpp`     
* `FusionEKF.cpp` - initializes the filter, calls the `predict` function, calls the `update` function.  takes the sensor data and initializes variables and updates variables. The Kalman filter equations are not in this file. `FusionEKF.cpp` has a variable called `ekf_`, which is an instance of a `KalmanFilter class`. The `ekf_` will hold the matrix and vector values. Uses the `ekf_` instance to call the `predict and update` equations.    
* `kalman_filter.cpp`*- defines the predict function, the update function for lidar, and the update function for radar.   
* `tools.cpp`- function to calculate RMSE and the Jacobian matrix.  
 
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

