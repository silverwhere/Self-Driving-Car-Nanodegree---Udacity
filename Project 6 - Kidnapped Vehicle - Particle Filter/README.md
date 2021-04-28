# Project 6 - Kidnapped Vehicle - Particle Filter Localization

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


Overview
---

The project involves a vehicle that is kidnapped within a world that has no idea of its location but can see landmarks using GPS and LIDAR sensors. Utilizing data from initial GPS estimates and LIDAR data, I can use a particle filter based on the vehicle's reported observations of objects nearby to localize it and find it!  The particle filter will be given a map and some initial localization information (analogous to what a GPS would provide). At each time step, the particle filter will also get observation and control data. 
 
<p align="center">
<img width="600" height="350" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%206%20-%20Kidnapped%20Vehicle%20-%20Particle%20Filter/img/particle_filter.jpeg"
</p>  
  
In this project I will implement a 2 dimensional particle filter in C++. The particle filter will be given a map and some initial localization information (GPS). At each time step my filter will also get observation and control data.  The project involves the completion of the following [rubric](https://review.udacity.com/#!/rubrics/747/view). 

Dependencies
---

This project requires [C++](https://isocpp.org/) and the following C++ libraries installed:  
  
  
File Structure
---

* `main.cpp` -  This file runs the particle filter, measures the runtime calculate the weighted error at each timestep.  
    - Set the number of particles `num_particles` to draw.  
    - Set the control measurement uncertainty `signma_pos`  
    - Set the landmark measurment uncertainty `sigma_landmark`  
    - Reads in map data  
    - Reads in control data and;  
    - Reads in observation data for each timestep.  

* `particle_filter.cpp` - Contains all the member functions of the `ParticleFilter` class.  
    - `init` function takes as input GPS coordinates `double x` and `double y`,  initial heading estimate `theta` and an array of uncertainties for these measurements `std[]`.      Then will sample from random Gaussian distribution centered around these measurements to initialize all the particles.  All particle weights will be initialized to 1.          Refer to `particle_struct` in `particle_filter .h`  
  
    - `prediction` takes as input the amount of time between timesteps `delta_t` the velocity and yaw rate measurment uncertainties `std_pos[]`and the current `velocity` and     `yaw_rate` timestep measurments.  Using these measurements the function will update the particles position estimates and account for sensor noise by adding Gaussian noise.      Gaussian noise can be added by sampling from a Gaussian distribution with mean equal to the updated particle position and standard deviation equal to the standard              deviation of the measurements.

    - `dataAssociation` takes as input two vectors of `LandmarkObs` objects; refer to `helpers_fuctions .h` for definition of this struct.  
    `vector<LandmarkObs> predicted` is the first vector which is prediction measurements between one particular particle and all of the map landmarks within sensor range `vector<LandmarkObs>& observations` is the actual landmark measurments gathered from the LIDAR Sensor.
     This function will perform nearest neighbour data association and assign each sensor observation the map landmark ID associated with it.  
      
    - `updateWeights` takes as input the range of the sensor `sensor_range` the landmark measurement uncertainties `std_landmark[]` a vector of landmark measurements                 `vector<LandmarkObs> observations` and `map_landmarks` as input.    
    
    - `resample()` function, use the weights of the particles in the particle filter and C++ standard libraries `discrete_distribution` function to update the particles to a          Bayesian posterior distribution.  

    - `weighted_mean_error` evalulates the performance of the particle filter by calculating the weighted error.  This function takes as input the ground-truth position `gt_x`        and `gt_y` at a particular timestep.    

Algorithim
---

**Implementation of Particle Filter**

**Initialization:**  

Initializes particle filter by initializing particles (num_particles = 100) to Gaussian distribution around first position and all the weights to 1.  The particles are initialized with a GPS position and heading. Every feasible position on the grid is called a particle and it represents a likely position of the vehicle inside the GPS location. 

* x Initial x position [m] (simulated estimate from GPS)
* y Initial y position [m]
* theta Initial orientation (heading) [rad]
* std[ ] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m], standard deviation of yaw [rad]]

**Prediction:**  
  
In more technical term each particle is a prior belief or simply a prior about the location and orientation of the vehicle. To represent different orientations or headings at the same location inside the GPS spot, one has to stack many such layers of particles, one for every possible orientation into a three-dimensional cube or cylinder.
Particles are more fine-grained priors, derived from a single, coarse prior provided by a GPS. With the priors set in place, the next step is to improve the belief or evaluate the priors against reality by sensing the environment.  

The location of each particle at the next time step after the time `delta_t` is predicted using the following calculation. The calculation depends on whether the heading (yaw rate) equals 0 or not:  

<p align="center">
<img width="1200" height="350" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%206%20-%20Kidnapped%20Vehicle%20-%20Particle%20Filter/img/motion_formulae.png"
</p>  

**Update Weights:**  

The vehicle uses LIDAR to sense its distance to landmarks, buildings, trees (observation measurements). LIDAR data is received as a list of x, y coordinates along with sensor noise mapped as a standard deviation in x and y. As the LIDAR sensor is attached to the vehicle, the measurements are in the vehicle's own coordinate system and not the coordinate system of the map which we will need to correctly perform observation measurement transformations, along with identifying measurement landmark associations to correctly calculate each particle's weight.

These observation measurements are transformed from vehicle coordinates (local coordinate system) to map coordinates (global coordinate system) using the following homogenous transformation matrix.

<p align="center">
<img width="800" height="350" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%206%20-%20Kidnapped%20Vehicle%20-%20Particle%20Filter/img/homogenous_transformation.png"
</p>  
  
<p align="center">
<img width="600" height="600" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%206%20-%20Kidnapped%20Vehicle%20-%20Particle%20Filter/img/observations.png"
</p>
                          
For each LIDAR observation (x, y) the nearest particle to a detected landmark is desired, a technique known as 'nearest neighbour' is utilized to perform this technique.  To accomplish this we use our transformed observation data coordinates from our LIDAR data.  Rarely is sensor data aligned perfectly with a map of known landmark coordinates and therefore there is a deviation.  Therefore each observation particle with the smallest deviation from the map of known landmark coordinates is determined by filtering all particles to determine the best fitting particle.

Associating the particle's pose with an observation (x,y), you land at some place. From this place you look for the nearest landmark on the map and associate this observation to a landmark. There is usually a difference between this observed location and the location of the landmark on the map. This deviation results in the so-called multivariate Gaussian probability. 

For each observation, the multivariate Gaussian normal distribution with its closest landmark is calculated and all resulting probabilities are multiplied. The result is the particle weight.  This is calculation is filtered through each particle.  The purpose is to have the particles with the highest weights survive and with each resampling, replacement particles are generated to remove particles with low posterior probability.

<p align="center">
<img width="300" height="150" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%206%20-%20Kidnapped%20Vehicle%20-%20Particle%20Filter/img/multivariate_gaussian_density.png"
</p>

All of the above has related to one single particle. Now the calculations are carried out for each particle. The particles with the highest weights gradually prevail in the algorithm (filtering). Hence, at the end of each weight update step, 'resampling' of particles with replacement is done to remove particles with low posterior probability.

**Resampling:**  

Weight disparity leading to weight collapse is a common issue encountered in these filtering algorithms; however, it can be mitigated by including a resampling step before the weights become too uneven.  Resampling involves keeping the particles with weights with a high posterior probability and removing those that do not.  This leads to more accurate particles with each successive update step. 

Results
---

Remarks
---
