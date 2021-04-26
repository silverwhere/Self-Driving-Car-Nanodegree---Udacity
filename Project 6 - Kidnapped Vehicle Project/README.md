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

* `main.cpp` -  This file runs the particle filter, measures the runtime calculate the weighted error at each timestep.  
             -  Set the number of particles `M` to draw.  
             -  Set the control measurement uncertainty `signma_pos`  
             -  Set the landmark measurment uncertainty `sigma_landmark`  
             -  Reads in map data  
             -  Reads in control data and;  
             -  Reads in observation data for each timestep.  



