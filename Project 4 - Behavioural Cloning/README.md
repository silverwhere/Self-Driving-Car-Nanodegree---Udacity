# Project 4 - Behavioural Cloning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

In this project I utilized end-to-end deep learning using convolutional neural networks (CNNs) to map the raw pixels from (3) a front-facing cameras to the steering commands for a self-driving car.  A simulator was used to capture the images during training laps around a track with various turns, curb styles, heights and pavement.  This powerful end-to-end approach means that with minimum training data from humans, the system learns to steer, with or without lane markings on roads.

Dependencies
---

This project requires Python 3.9 and the following Python libraries installed:  
  
[Keras](https://keras.io/)  
[TensorFlow](https://www.tensorflow.org/)  
[OpenCV](https://opencv.org/)  
  

Data Collection
---

Data collection was performed by driving a car in a simulator utilzing keyboard and mouse commands for steering and throttle.  To collect data I drove the car along the centre of the road for a single lap and recorded all of the images.  I then drove the car along the centre-line in reverse on the same track for a single lap.  Finally, to capture some edge cases, I wanted to steer close to borders of the road on turns, areas with no curb and walls on a bridge, returning to centre as soon as I got to close.  Steering angles for each image were captured via `.csv` driving log; images were captured for a left, centre and right camera `160 x 320 x 3`  as follows.  
  
<p align="center">
<img width="500" height="310" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%204%20-%20Behavioural%20Cloning/examples/training.png"</p>  
  
Data Augmentation
---  

Network Architecture
---    
  
Most of the popularized well-known CNNs use `classification`, but this has to do with the fact that more often than not, in both both machine learning and deep learning, we’re most interested to give a specific label to something, and the training data itself comes in this format, i..e, as in [Project 3 - Traffic Sign Classifier](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/tree/main/Project%203%20-%20Traffic%20Sign%20Classifier) it comes as a list of pairs of examples of (images, label).

The most common loss function for classification in this context is cross-entropy loss, summed across all your output nodes. This has to do with the fact that we are comparing a set of probabilities output by the network with a bunch of 0’s and 1 (or a few more than 1 in the case of multi-label classification) “1”s to indicate the one-hot encoded labels that apply for each label.   

However, with `regression`, things are a little different, because the network is trained not on specific labels, but on training examples such as steering angle paired with numeric steering angle values. This means we need a different, more appropriate loss function, with the most obvious choice being mean squared error loss. (Note; for both classification and regression, many choices are possible!).

Typically when we hear “regression”, we mean we want to predict a single numeric independent variable. For this CNN we will use regression to output the correct steering angle for our position on the road.  To accomplish this, we will utilize Keras!  Keras makes coding deep neural networks simpler.  

![CNN](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%204%20-%20Behavioural%20Cloning/examples/cnn-architecture.png)

  

