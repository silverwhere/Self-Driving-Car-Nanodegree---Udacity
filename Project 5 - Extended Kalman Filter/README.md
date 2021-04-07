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
* `kalman_filter.cpp`* - defines the `predict` function, the `update` function for lidar, and the `update` function for radar.   
* `tools.cpp`- function to calculate `RMSE` and the `Jacobian matrix`.  

Process Flow
---

* We have a pedestrian on a bicycle that is represented by a 2-D position `P<sub>x</sub>`, `P<sub>y</sub>`  as well as 2-D velocity `v<sub>x</sub>`, `v<sub>y</sub>` 

Algorithim
---

The Kalman Filter algorithm will go through the following steps:  

**first measurement** - the filter will receive initial measurements of the bicycle's position relative to the car. These measurements will come from a radar or lidar sensor.

---
**initialize state and covariance matrices** - the filter will initialize the bicycle's position based on the first measurement. Then the car will receive another sensor measurement after a time period Δt.

---

**predict (state prediction)** - the algorithm will predict where the bicycle will be after time Δt. One basic way to predict the bicycle location after Δt is to assume the bicycle's velocity is constant; thus the bicycle will have moved velocity Δt.  We will assume the velocity is constant. 
  
Let’s say we know an object’s current position and velocity , which we keep in the x variable. Now one second has passed. We can predict where the object will be one second later because we knew the object position and velocity one second ago; we’ll just assume the object kept going at the same velocity.
  
But maybe the object didn’t maintain the exact same velocity. Maybe the object changed direction, accelerated or decelerated. So when we predict the position one second later, our uncertainty increases.
  
**x** is the mean state vector. For an extended Kalman filter, the mean state vector contains information about the object’s position and velocity that you are tracking. It is called the “mean” state vector because position and velocity are represented by a gaussian distribution with mean x.  

**P** is the state covariance matrix, which contains information about the uncertainty of the object’s position and velocity. You can think of it as containing standard deviations.  

**Q**  is the Process Covariance Matrix. It is a covariance matrix associated with the noise in states. I didn’t specify the equation to calculate the matrix here.

**F** is the Transition Matrix (the one that deals with time steps and constant velocities) 

---

![Figure](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%205%20-%20Extended%20Kalman%20Filter/img/kalmanequations.png)

---
**update (measurement update)** - the filter compares the "predicted" location with what the sensor measurement says. The predicted location and the measured location are combined to give an updated location. The Kalman filter will put more weight on either the predicted location or the measured location depending on the uncertainty of each value.  Then the car will receive another sensor measurement after a time period Δt. The algorithm then does another predict and update step.

**z** is the measurement vector.   
* For a lidar sensor, the z vector contains the position−x and position−y measurements.
* For a radar sensor, the z vector contains the range(rho), bearing(phi) and radial velocity(rho_dot).

**H**  is the matrix that projects your belief about the object’s current state into the measurement space of the sensor.  

* For lidar, this is a fancy way of saying that we discard velocity information from the state variable since the lidar sensor only measures position: The state vector x contains information about [px​,py​,vx​,vy​] whereas the z vector will only contain [px,py]. Multiplying Hx allows us to compare x, our belief, with z, the sensor measurement. 
* For radar, there is no H matrix that will map the state vector "x" into polar cordinates; instead you need to calculate the mapping manually to convert from cartesian coordinates to polar coordinates.  

The **H** matrix from the Lidar and **h(x)** equations from Radar are accomplishing the same thing;  they are both need to solve **y = Z - H*x'** 

![Figure](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%205%20-%20Extended%20Kalman%20Filter/img/radar%20equations.png)


**R** is the covariance matrix.  For a radar sensor this matrix represents uncertainty in our sensor measurments.  The dimensions of the R matrix is squared and each side of its matrix is the same length as the number of measurement parameters.

* For laser sensors, we have a 2D measurement vector.  Each location component Px, PY are affected by a random noise.  So our noise vector "w" has the same dimension as "z".  And it is a distribution with zero mean and a 2x2 covariance matrix which comes from the product of the vertical vector "w" and its transpose.  

***Extended Kalman Filter / Sensor Fusion***  

Extended Kalman Filters(EKF) linearize the distribution around the mean of the current estimate and then use this linearization in the predict and update states of the Kalman Filter algorithm.  An existing Kalman Filter cannot be applied to a non-linear distribution, common with Radar data.  The key to be able to solve the kalman filter update equations is to linearize the **h(x)** function.

<p align="center">
<img width="800" height="250" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%205%20-%20Extended%20Kalman%20Filter/img/gaussian.png"
</p>


As the **h(x)** function is multi-dimensional the EKF will need to utilize a method called a multivariable Taylor Series Expansion to make a linear approximation of the **h(x)** function.   

![Figure](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%205%20-%20Extended%20Kalman%20Filter/img/taylor.jpg) 
  
where **Df(a)** is called the `JacobianMatrix` & **D<sup>2</sup>f(a)** is called the Hessan matrix.  These represent the first and second order derivatives of multi-dimensional equations.

**Jacobian Matrix** - To derive a linear approximations for **h(x)** function, we will only keep the expansion up to the `JacobianMatrix` **Df(a)**.  We will ignore the Hessan matrix **D<sup>2</sup>f(a)** and other higher order terms.  Assuming **(x-a)** is small, **(x-a)<sup>2</sub>** or the multi-dimensional equivalent **(x-a)<sup>T</sup>(x-a)** will be even smaller; the EKF we'll be using assumes that higher order terms beyond the Jacobian are negligible.  


---  
Remarks
---

