# Project 4 - Behavioural Cloning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)  

Overview
---

In this project, I utilized end-to-end deep learning using convolutional neural networks (CNNs) to map the raw pixels from (3) front-facing cameras to the steering commands for a self-driving car. A simulator was used to capture the images during training laps around a track with various turns, curb styles, heights and pavement. This powerful end-to-end approach means that with minimum training data from humans, the system learns to steer, with or without lane markings on roads.

<p align="center">
<img width="500" height="250" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%204%20-%20Behavioural%20Cloning/behavioral_cloning_gif.gif"
</p>



Dependencies
---

This project requires [Python](https://www.python.org) and the following Python libraries installed:  
  
[Keras](https://keras.io/)  
[TensorFlow](https://www.tensorflow.org/)  
[OpenCV](https://opencv.org/)  
  

Data Collection
---

Data collection was performed by driving a car in a simulator utilizing keyboard and mouse commands for steering and throttle. To collect data I drove the car along the centre-line of the road for two laps and recorded all of the images. I then drove the car along the centre-line in reverse on the same track for a single lap. Finally, to capture some edge cases, I wanted to steer close to borders of the road on turns, areas with no curb and walls on a bridge, returning to the centre as soon as I got to close. For each frame of the image, the feature or image was stored in a file & the corresponding label measurement for the steering angle was also captured via a driving log file `.csv`; `features` and `labels` were captured for a left, center and right positioned camera `160 x 320 x 3`  as illustrated.    
  
<p align="center">
<img width="800" height="510" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%204%20-%20Behavioural%20Cloning/examples/diagram.jpg"</p>  

Data Augmentation
---  
After saving my data, I wanted to increase the amount of my data, therefore I `flipped` each frame of the video along the vertical axis, and correspondingly multiplied the ground truth measurement by -1 to correct for the flipped position.  

Additionally, using a `Cropping2D Layer` I was able to crop the images 75 pixels from the top of each frame and 20 pixels from the bottom. This was done to increase the performance time for the model to focus only on the areas that require training for the steering angle. As can be seen from the example image below, removing the area above the road with a background such as a sky, terrain and the hood of the car can aid in the model training more accurately and faster.  

  
<p align="center">
<img width="500" height="400" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%204%20-%20Behavioural%20Cloning/examples/original-image.jpg"</p>
 

Network Architecture
---    
  
Most of the popularized well-known CNNs use `classification`, but this has to do with the fact that more often than not, in both machine learning and deep learning, we’re most interested to give a specific label to something, and the training data itself comes in this format, i..e, as in [Project 3 - Traffic Sign Classifier](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/tree/main/Project%203%20-%20Traffic%20Sign%20Classifier) it comes as a list of pairs of examples of (images, label).

The most common loss function for classification in this context is a cross-entropy loss, summed across all your output nodes. This has to do with the fact that we are comparing a set of probabilities output by the network with a bunch of 0’s and 1 (or a few more than 1 in the case of multi-label classification) “1”s to indicate the one-hot encoded labels that apply for each label.   

However, with `regression`, things are a little different, because the network is trained not on specific labels, but on training examples such as steering angle paired with numeric steering angle values. This means we need a different, more appropriate loss function, with the most obvious choice being mean squared error loss. (Note; for both classification and regression, many choices are possible!).

Typically when we hear “regression”, we mean we want to predict a single numeric independent variable. For this CNN we will use regression to output the correct steering angle for our position on the road.  To accomplish this, we will utilize Keras!  Keras makes coding deep neural networks simpler.  

![CNN](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%204%20-%20Behavioural%20Cloning/examples/cnn-architecture.png)Source: [NVIDIA](https://arxiv.org/pdf/1604.07316v1.pdf)  

My goal for this project was to build my regression CNN to output a single output node to predict the steering angle required for a given position on the test track.  To start, I utilized a CNN Architecture developed by NVIDIA as seen above.  When running the initial model, I achieved a great result in that my car was driving autonomously!  But in reality, it was swerving to the left and the right of the frame rhythmically.  To improve my model, I added layers such as a `max pooling` layer after most 2D convolutions, and a dropout layer to reduced overfitting. 
  

A `convolutional layer` that extracts features from a source image. Convolution helps with blurring, sharpening, edge detection, noise reduction, or other operations that can help the machine to learn specific characteristics of an image.  

A `pooling layer` that reduces the image dimensionality without losing important features or patterns.    
  
A `Flatten layer` is in between the `convolutional layer` and the `fully connected layer`. Flattening transforms a two-dimensional matrix of features into a vector that can be fed into a fully connected neural network classifier. 

A `dropout layer` is a regularization technique for reducing overfitting.  The technique temporarily drops units (artificial neurons) from the network, along with all of those units' incoming and outgoing connections.
  
A fully connected layer also known as the `dense layer`, in which the results of the convolutional layers are fed through one or more neural layers to generate a prediction.  

Below is a network summary from Keras/TensorFlow displaying my model used.  Note that initially, I added only a single `dropout layer` at the top of the network, however, upon training with eventually what I deemed to be good data, I decided to apply one very early after the first `convolutional layer` to see if the model would improve, and it did!  The model can be viewed in `model.py`  

<p align="center">
<img width="600" height="810" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%204%20-%20Behavioural%20Cloning/examples/CNN_Final.jpg"</p>   
  
Training
---    
  
The training involved utilizing the Udacity Vehicle Simulator.  While the simulator was not based on a real vehicle, the physics engine and use of cameras to detect features in our recorded image space would be applicable in the real world; therefore this is a very valuable project.  The training proved to be the most time-consuming aspect of this project for me.  While I was limited in the functionality of my computer `mouse` to adjust steering angle, one fact holds.  That fact is that *Garbage Data In = Garbage Data Out*, or in my case poor prediction of steering angle.  During training, if I drive off the road slightly and correct for repetitive areas on the track, there will be enough training data for the CNN to learn and predict the correct state.  However, with bad data combined with data augmentation to create additional data, those bad data points now occur at a higher magnitude of occurrence and thus will have an impact on the overall training of my CNN.  To solve this, I realized that while I could keep adding more additional, good data points, it was better for me to start fresh, with a smaller, cleaner data set.  Therefore, my final model training data is based on 3 laps in the forward direction, and 1 lap in reverse.  
  
After running my CNN, I noticed in viewing my autonomous driving though that a few areas, especially where the curb disappears and in turns, the car would have trouble navigating those areas.  As my CNN was performing well, I attributed this to a limited amount of turning data, therefore, I added several more recordings in the areas of those turns.

Mean Squared Error Loss / Adam Optimizer
---

I utilized the following loss function and optimizer while training my network.  

The `Mean Squared Error, or MSE`, loss is the default loss to use for regression problems.

Mean squared error is calculated as the average of the squared differences between the predicted and actual values. The result is always positive regardless of the sign of the predicted and actual values and a perfect value is 0.0. The squaring means that larger mistakes result in more error than smaller mistakes, meaning that the model is punished for making larger mistakes.  

For my model, after 5 `EPOCHS` I noticed that the `MSE` after each `EPOCH` for the validation data correlated well against the training set.  

![MSE](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%204%20-%20Behavioural%20Cloning/mean_squared_error_loss.png)  

Optimizer - `Adam` is a replacement optimization algorithm for stochastic gradient descent (SGD) for training deep learning models. Adam combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems.

Autonomous Driving
---

Once my model was validated I ran my model utilizing `drive.py` and `model.h5` file generated by Keras.   An `.h5` file easily stores the weights and model configuration in a single file. Upon launching the simulator the vehicle began to accelerate to a set speed and was able to navigate the test track quite smoothly with no error.  This autonomous driving was performed by visualizing `features` from a single camera positioned on the center of the vehicle, utilizing the trained behavioural cloning network to predict the correct steering angle `label` for a given vehicle position in a lane.  A video of the performance of this model was recorded using `video.py` from each of the individual frames stored while running my model.  Additionally, a video `run1.mp4` was created to show the vehicle navigating around the test track.    

<p align="center">
<img width="700" height="150" src="https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%204%20-%20Behavioural%20Cloning/examples/network_computed.png"</p>  
  
Remarks
---

Behavioural Cloning was a great project to truly learn how vehicle systems can learn to drive a vehicle autonomously.  While this project doesn't capture every edge case one could expect in the real world, what it has shown though is how powerful deep-learning is to regressively determine the desired output.  What would have been interesting in this project would have also been to log the vehicle speed.  In reality, you would probably want to go faster on straightaways with minimal steering angle and slower on turns with larger positive or negative steering angles instead of a constant speed throughout.   









  

