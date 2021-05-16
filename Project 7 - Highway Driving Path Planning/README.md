# Project 7 - Highway Driving Path Planning - Trajectory Generation

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


Overview
---
    
In this project my goal is to safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. I will utilize the car's localization and sensor fusion data, there is also a sparse map list of waypoints around the highway. There are several parameters required to successfully pass the project.  The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, note that other cars will try to change lanes too. The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another. The car should be able to make one complete loop around the 6,946m (4.32 mile) highway. Since the car is trying to go 50 MPH, it should take a little over 5 minutes to complete 1 loop. Also the car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3, as this would affect passenger comfort.  A detailed list of the projects requiremtns can be found in the project [Rubic](https://review.udacity.com/#!/rubrics/1971/view)    

Simulator
---  

You can download the Term3 Simulator which contains the Path Planning Project from the [releases tab (https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2).  

To run the simulator on Mac/Linux, first make the binary file executable with the following command:
```shell
sudo chmod u+x {simulator_file_name}
```

Dependencies
---

* This project requires [C++](https://isocpp.org/) and the following C++ libraries installed:    
  * spline.h  
* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ``  
Map & Localization Data Variables for Sensor Fusion and Localization
---  

The map of the highway is in `data/highway_map.txt`
Each waypoint in the list contains  `[x,y,s,dx,dy]` values. `x` and `y` are the waypoint's map coordinate position, the `s` value is the distance along the road to get to that waypoint in meters, the `dx` and `dy` values define the unit normal vector pointing outward of the highway loop.

The highway's waypoints loop around so the frenet `s` value, distance along the road, goes from 0 to 6945.554.


Here is the data provided from the Simulator to the C++ Program

#### Main car's localization Data (No Noise)

["x"] The car's x position in map coordinates

["y"] The car's y position in map coordinates

["s"] The car's s position in frenet coordinates

["d"] The car's d position in frenet coordinates

["yaw"] The car's yaw angle in the map

["speed"] The car's speed in MPH

#### Previous path data given to the Planner

//Note: Return the previous list but with processed points removed, can be a nice tool to show how far along
the path has processed since last time. 

["previous_path_x"] The previous list of x points previously given to the simulator

["previous_path_y"] The previous list of y points previously given to the simulator

#### Previous path's end s and d values 

["end_path_s"] The previous list's last point's frenet s value

["end_path_d"] The previous list's last point's frenet d value

#### Sensor Fusion Data, a list of all other car's attributes on the same side of the road. (No Noise)

["sensor_fusion"] A 2d vector of cars and then that car's [car's unique ID, car's x position in map coordinates, car's y position in map coordinates, car's x velocity in m/s, car's y velocity in m/s, car's s position in frenet coordinates, car's d position in frenet coordinates. 

Details
---  

The car uses a perfect controller and will visit every (x,y) point it recieves in the list every .02 seconds. The units for the (x,y) points are in meters and the spacing of the points determines the speed of the car. The vector going from a point to the next point in the list dictates the angle of the car. Acceleration both in the tangential and normal directions is measured along with the jerk, the rate of change of total Acceleration. The (x,y) point paths that the planner recieves should not have a total acceleration that goes over 10 m/s^2, also the jerk should not go over 50 m/s^3. (NOTE: As this is BETA, these requirements might change. Also currently jerk is over a .02 second interval, it would probably be better to average total acceleration over 1 second and measure jerk from that.

There will be some latency between the simulator running and the path planner returning a path, with optimized code usually its not very long maybe just 1-3 time steps. During this delay the simulator will continue using points that it was last given, because of this its a good idea to store the last points you have used so you can have a smooth transition. previous_path_x, and previous_path_y can be helpful for this transition since they show the last points given to the simulator controller with the processed points already removed. You would either return a path that extends this previous path or make sure to create a new path that has a smooth transition with this last path.

Results
---

My path planner was able to correctly navigate the highway for greater than the requirement of 4.32 miles, being able to both maintain my lane and change lanes correctly, not over accelerate and maintain the desired target speed of 50 MPH.  By utilizing a spline from `spline.h` I was able to effectively create a smooth trajectory when changing lanes, which minimized acceleration normal to the vehicle `AccN`.  Further, by adding functional safety logic to my path planner I was able to utilize the localization of my vehicle and with sensor fusion the vehicles around me to determine if it was safe to change lanes safely or brake for vehicles ahead to avoid all collisions.  While my estimates were conservative at 50 metres for a minimum safe distance for changing lanes; I believe if I was not restricted to acceleration `Acc` < 10 m/s^2 and `Jerk` < 10 m/s^3 I believe I could have lapped the track faster.  

Remarks
--- 

The project was extremely interesting and I can certainly see my knowledge across the whole nanodegree converging into trajectory generation.  As I review videos being performed by one company named [Cruise](https://www.getcruise.com/) I really have a strong understanding of path planning and also an great appreciation of all the details that have to be considered when planning a trajectory.  Furthermore, of interest are edge cases, such as dynamic events such as temporary construction on routes utilizing localization data with provided maps.  Probably a good reason to consider other alternatives such as SLAM to aid in these events.


References
--- 

A really helpful resource for doing this project and creating smooth trajectories was using http://kluge.in-chemnitz.de/opensource/spline/, the spline function is in a single hearder file is really easy to use.  I utilized the spline.h library to help fit a smooth curve for my trajectory over a distance with `N` trajectory points evenly spaced to control my vehicles acceleration and velocity.

---


