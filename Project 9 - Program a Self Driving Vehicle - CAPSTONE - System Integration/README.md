# Project 9 - System Integration - Programming a Real Self Driving Vehicle

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


Overview
---

Udacity Nanodegree CAPSTONE!  This project is the culmination of the work learned to date in the nanodegree.  In this project, I will be responsible to program several subsystems to operate "CARLA" the Udacity Self Driving Car, both in the simulator and on-road.  In the perception subsystem, I will implement traffic light detection.  In the planning subsystem, I will implement a waypoint updater based on traffic light status.  In the control subsystem, I will be implementing drive-by-wire nodes that take target trajectory information as input and send control commands to control the vehicle.  The project will be developed on a simulator and then can be tested on a real vehicle!
    
[![YouTube](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%209%20-%20Program%20a%20Self%20Driving%20Vehicle%20-%20CAPSTONE%20-%20System%20Integration/imgs/YouTube.jpg)](https://youtu.be/HeZ4mnrke7I)  

ROS Architecture
---  
For this project, I wrote ROS nodes to implement core functionality of the autonomous vehicle system, including traffic light detection, control, and waypoint following! I tested my code using a simulator that mimics the functionality of Carla prior to being run on the real vehicle.  Note: All vehicle parameters (i.e. wheel radius, vehicle mass, fuel tank capacity) match those of the real Udacity vehicle for accurate simulation.

The following is a system architecture diagram showing the ROS nodes and topics used in the project.  The ROS nodes and topics shown in the diagram are described briefly in the Code Structure section below, and more detail is provided for each node in later classroom concepts of this lesson.  

![Ros Arch](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%209%20-%20Program%20a%20Self%20Driving%20Vehicle%20-%20CAPSTONE%20-%20System%20Integration/imgs/ros_architecture.png)  

Code Structure
---

The code is contained entirely within the `/ros/src/` directory. Within this directory, you will find the following ROS packages:

### /ros/src/tl_detector/  

This package contains the traffic light detection node: `tl_detector.py`. This node takes in data from the `/image_color`, `/current_pose`, and `/base_waypoints` topics and publishes the locations to stop for red traffic lights to the `/traffic_waypoint topic`.  

The `/current_pose` topic provides the vehicle's current position, and `/base_waypoints` provides a complete list of waypoints the car will be following.  

I built both a traffic light detection node and a traffic light classification node. Traffic light detection takes place within `tl_detector.py`, whereas traffic light classification takes place within `../tl_detector/light_classification_model/tl_classfier.py`.  

![tl_detector](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%209%20-%20Program%20a%20Self%20Driving%20Vehicle%20-%20CAPSTONE%20-%20System%20Integration/imgs/tl-detector-ros-graph.png)  

### /ros/src/waypoint_updater/  

This package contains the waypoint updater node: `waypoint_updater.py`. The purpose of this node is to update the target velocity property of each waypoint based on the traffic light and obstacle detection data. This node will subscribe to the `/base_waypoints`, `/current_pose`, `/obstacle_waypoint`, and `/traffic_waypoint` topics, and publish a list of waypoints ahead of the car with target velocities to the `/final_waypoints` topic.  

![waypoint_updator](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%209%20-%20Program%20a%20Self%20Driving%20Vehicle%20-%20CAPSTONE%20-%20System%20Integration/imgs/waypoint-updater-ros-graph.png)   

### /ros/src/twist_controller/  
Carla is equipped with a drive-by-wire (dbw) system, meaning the throttle, brake, and steering have electronic control. This package contains the files that are responsible for control of the vehicle: the node `dbw_node.py` and the file `twist_controller.py`, along with a PID and lowpass filter that you can use in your implementation. The dbw_node subscribes to the `/current_velocity` topic along with the `/twist_cmd` topic to receive target linear and angular velocities. Additionally, this node will subscribe to `/vehicle/dbw_enabled`, which indicates if the car is under dbw or driver control. This node will publish throttle, brake, and steering commands to the `/vehicle/throttle_cmd`, `/vehicle/brake_cmd`, and `/vehicle/steering_cmd` topics.  

![twist_controller](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%209%20-%20Program%20a%20Self%20Driving%20Vehicle%20-%20CAPSTONE%20-%20System%20Integration/imgs/dbw-node-ros-graph.png)  

In addition to these packages you will find the following, which are not necessary to change for the project. The `styx` and `styx_msgs` packages are used to provide a link between the simulator and ROS, and to provide custom ROS message types:

### /ros/src/styx/
A package that contains a server for communicating with the simulator, and a bridge to translate and publish simulator messages to ROS topics.

### /ros/src/styx_msgs/
A package that includes definitions of the custom ROS message types used in the project.

### /ros/src/waypoint_loader/
A package which loads the static waypoint data and publishes to `/base_waypoints`.

### /ros/src/waypoint_follower/
A package containing code from Autoware which subscribes to `/final_waypoints` and publishes target vehicle linear and angular velocities in the form of twist commands to the `/twist_cmd topic`.

Details
---  

I. Perception

Utilizing a camera the Udacity self-driving car "Carla" can use machine learning and deep learning methods to train a traffic light classifier to predict the traffic light state. In the simulator as long as the camera sees a traffic light, it can evaluate whether it is Red, Yellow or Green. I choose to obey traffic laws, so when if we see a yellow light for instance we will continue through the intersection and not speed up. A red light will require the vehicle to detect the red light and to begin a gentle deceleration. Green light will mean proceeding through the intersection.   
   
II. Planner

We can think of our waypoint planner as a HD Map with x, y coordinates easily determined. These pre-loaded coordinates of continuous points are what we define as 'waypoints'. The path planner will use these waypoints to determine the acceleration necessary to maintain our desired target velocity between each waypoint line segment. Further, we will need to update our target velocity when information from perception subsystems, such as a Red or Green light are detected. If a Red light is detected we will want our vehicle to slow down and stop at the stop line of the intersection with the detected redlight. Calculating our distance between our current waypoint and target stopping waypoint we can calculate a deceleration curve to ensure our velocity decreases as we get closer to the intersection. Note we will need to apply a braking torque ~ 700 Newton-metres to hold the vehicle in place while stopped at the intersection.  

The planner will plan all actions for the car, assigning target velocities (angular and linear) to every waypoint ahead of the vehicle in the planned trajectory ahead.
  
III. Controls

Based on the inputs from the planner we can control the vehicle's steering, throttle and braking. The goal of the vehicle is to follow the waypoints as planned. To prevent major oscillation or deviation from the path a PID controller is utilized to correctly operate the vehicle along the desired trajectory. By monitoring any error from our actual trajectory to the planned waypoint trajectory the PID controller can apply steering inputs.  
  
Further, based on input from perception and planning, if an intersection is detected with a green or red light, or a red light that turns green after already decelerating, the controller can accelerate/decelerate the vehicle at a comfortable rate of acceleration. Current velocity and target velocity will be monitored and corrected should the current velocity be too high or low from the target velocity.


Results
---

The results of my ROS implemented code is a vehicle that is able to navigate the simulator without issue.  

* The vehicle is able to accelerate and deceleration within desired ranges.  
* The vehicle is able generate waypoints and navigate each of them maintaining a target velocity
* Steering is maintained with oscillation minimized with the use of a PID controller
* Object detection of traffic lights is achieved with the vehicle able to navigate stopping or proceeding through the interesection without issue.


Remarks
--- 

This is my final remarks on what has been an almost 2 year journey beginning with Intro to Self Driving Cars!  I look back fondly at all the math and programming I had to learn to achieve this final project and have what I consider to be an amazing understanding of the workings of a self driving vehicle.  Overall, I really enjoyed the final project.  Learning how to actuate a vehicle using ROS was very informative.  As I understand the simulator was designed to be implemented in Carla, perhaps this is why traffic was not added to the simulator as in past projects.  Overall though I am very excited to share that I have developed software that has been used on a real self-driving vehicle.

Thank you Sebastian Thrun & David Silver for what I consider to be a wonderful complement to my Mechanical Engineering skillset!  

---



This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

We are working on a fix to line up the OpenCV versions between the two.
