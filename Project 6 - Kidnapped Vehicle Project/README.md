# Project 6 - Kidnapped Vehicle

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project I will implement a 2 dimensional particle filter in C++. The particle filter will be given a map and some initial localization information (GPS). At each time step my filter will also get observation and control data.  The project involves the completion of the following [rubric](https://review.udacity.com/#!/rubrics/747/view).
  

Overview
---

<p align="center">
<img width="600" height="350" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%205%20-%20Extended%20Kalman%20Filter/img/overview.jpeg"
</p>


Dependencies
---

This project requires [C++](https://isocpp.org/) and the following C++ libraries installed:  
  

  
File Structure
---

* `main.cpp` -  communicates with a simulator that recieves Lidar and Radar data from our vehicle, calls a function to run the kalman filter and calls a function to calculate the error RMSE.  Sends a sensor measurement to `FusionEKF.cpp`     
* `FusionEKF.cpp` - initializes the filter, calls the `predict` function, calls the `update` function.  takes the sensor data and initializes variables and updates variables. The Kalman filter equations are not in this file. `FusionEKF.cpp` has a variable called `ekf_`, which is an instance of a `KalmanFilter class`. The `ekf_` will hold the matrix and vector values. Uses the `ekf_` instance to call the `predict and update` equations.    
* `kalman_filter.cpp`* - defines the `predict` function, the `update` function for lidar, and the `update` function for radar.   
* `tools.cpp`- function to calculate `RMSE` and the `JacobianMatrix`.  


