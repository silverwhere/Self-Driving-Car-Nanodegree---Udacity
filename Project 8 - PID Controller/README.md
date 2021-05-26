# Project 8: PID Control Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
PID stands for Proportional-Integral-Derivative. These three components are combined in such a way that it produces a control signal to steer a vehicle.  In this project I developed a PID controller in C++ to maneuver a vehicle around a track in the Udacity simulator. The simulator will provide the cross track error (CTE) and the velocity (mph), while my algorithim will be developed to steer and maneuver the vehicle around the test track with minimal oscillation.

[![YouTube](https://youtu.be/WaVhuAvrYF0)]

## Cross Track Error  
A cross track error is distance between the vehicle´s actual trajectory and the intended trajectory. In theory it’s best suited to control the car by steering in proportion to Cross Track Error (CTE).  For this project the CTE is measured against the centreline of the lane, however in other cases it may be measured from the edge of the lane, or lane line.  
  
## P component
It sets the steering angle in proportion to CTE with a proportional factor `tau`.  (the coefficient `tau` is called "response strength"):
  
```steering angle = -tau_p * cte```  

This term applies a correction to the steering wheel proportional to the error. If we are too far from the goal, we turn the wheel in the other direction.
   
In other words, the P, or "proportional", component had the most directly observable effect on the car’s behaviour. It causes the car to steer proportional (and opposite) to the car’s distance from the lane center(CTE) - if the car is far to the right it steers hard to the left, if it’s slightly to the left it steers slightly to the right.  As can be seen from the figure below, the limitation of the P controller is that it will oscillate along the intended trajectory.
  
The disadvantage of a single P Controller is that it causes a constant oscillation. 

<p float="left">
  <img src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%208%20-%20PID%20Controller/img/p-ctrl.png" width="500" />
  <img src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%208%20-%20PID%20Controller/img/p.gif" width="500" height="320"/> 
</p>  
  
## D component  
It’s the differential component of the controller which helps to take temporal derivative of error. This means when the car turned enough to reduce the error, it will help not to overshoot through the x axis. In other words, the `D`, or "differential", component counteracts the `P` component’s tendency to ring and overshoot the center line. A properly tuned `D` parameter will cause the car to approach the center line smoothly without oscillation.  
```
diff_cte = cte - prev_cte
prev_cte = cte
steering angle = -tau_p * cte - tau_d * diff_cte
```  
The purpose of the term D is to suppress this oscillation effect by adding a damping term to the formula. This term is the change of error. The PD controller understands that the error decreases and slightly reduces the angle it adopts to approach a smooth path.   

<p float="left">
  <img src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%208%20-%20PID%20Controller/img/pd-ctrl.png" width="500" height="300" />
  <img src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%208%20-%20PID%20Controller/img/pd.gif" width="500" height="300"/> 
</p>  
  
    
## I component
It’s the integral or sum of error to deal with systematic biases such as mechanical error.  In other words, the `I`, or "integral", component counteracts a bias in the CTE which prevents the `P-D` controller from reaching the center line. This bias can take several forms, such as a steering drift , but I believe that in this particular implementation the `I` component particularly serves to reduce the CTE around curves.

```steering angle = -tau_p * cte - tau_d * diff_cte - tau_i * int_cte```    
  
<p align="center">
<img width="600" height="350" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%208%20-%20PID%20Controller/img/pid-ctrl.png"
</p>  

## Twiddle (Coordinate Ascent)  
Parameter optimisation can be done manually or using Twiddle algorithm.  Twiddle is an algorithm that tries to find a good choice of parameters 
p for our PID parameters that returns an error.  This is useful as each error from `p_error`, `i_error` and `d_error` all have a contributing effect on each other error respectively.  By tuning our parameters we are able to effectively find the choice of parameters based on the `best_err` for each parameter selected.  The result will be a vehicle that is able to utilize the PID controller with parameters that produce minimal error for each part of the controller.  A trade-off of Twiddle however is that the twiddle controller will initially overshoot the intended trajectory initially as seen from the output below.

<p align="center">
<img width="600" height="350" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%208%20-%20PID%20Controller/img/twiddle.jpg"
</p>    
  
---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

Fellow students have put together a guide to Windows set-up for the project [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/Kidnapped_Vehicle_Windows_Setup.pdf) if the environment you have set up for the Sensor Fusion projects does not work for this project. There's also an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3).

File Structure
---

* `main.cpp`   
    -   Initializes the PID variable with the tuned parameters
    -   Updates errors and calculate `steer_value` at each step
    -   Calculates new steering angle
         
* `PID.cpp` 
  -   Initializes the PID coefficients
  -   Initializes the errors for P, I and D
  -   Function to update errors for P, I and D
  -   Function to update total error

Results
---
  
I was able to successfully navigate around the test track with the parameters selected for my PID algorithim.  A video link to my results is posted:
  

https://youtu.be/WaVhuAvrYF0
