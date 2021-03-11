# Project 4 - Behavioural Cloning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

In this project I utilized end-to-end deep learning using convolutional neural networks (CNNs) to map the raw pixels from (3) a front-facing cameras to the steering commands for a self-driving car.  A simulator was used to capture the images during training laps around a track with various turns, curb styles, heights and pavement.  This powerful end-to-end approach means that with minimum training data from humans, the system learns to steer, with or without lane markings on roads.

Data Collection
---

Data collection was performed by driving a car in a simulator utilzing keyboard and mouse commands for steering and throttle.  To collect data I drove the car along the centre of the road for a single lap and recorded all of the images.  I then drove the car along the centre-line in reverse on the same track for a single lap.  Finally, to capture some edge cases, I wanted to steer close to borders of the road on turns, areas with no curb and walls on a bridge, returning to centre as soon as I got to close.  Images were captured for a left, centre and right camera as follows.  
  
<p align="center">
<img width="500" height="310" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%204%20-%20Behavioural%20Cloning/examples/training.png"</p>

