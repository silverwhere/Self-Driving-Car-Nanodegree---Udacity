# Project 3: Building a Traffic Sign Classifier (Recognition Program)
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I utilized deep neural networks and convolutional neural networks to classify traffic signs. I trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model was trained, I tested out the model on images of German traffic signs I found on the web.  The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements, of which I believe I accomplished by augmenting the training data and challenging my classifier with "toxic" difficult to read road signs.

The Project
---
The goals / steps of this project are the following:
* Load the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)  

* Explore, summarize and visualize the data set  
  
A sample of the data set is included below. From the visualization of a histogram we can clearly see this is a large amount of training data, however there is very little validation data. The test data seems appropriate enough to test our classifier on.  

![Figure 1](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%203%20-%20Traffic%20Sign%20Classifier/project_screenshots/training_data_sample.png)  

From the visualization of a histogram we can clearly see this is a large amount of training data, however there is very little validation data. The test data seems appropriate enough to test our classifier on. 

![Figure_2](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%203%20-%20Traffic%20Sign%20Classifier/project_screenshots/histogram.jpg)  

The histogram above shows an overlay of each data set across each of the 43 different sign classes. Of additional note is that there are limited images available for individual classes, which may make those images harder to classify accurately. I will want to use data augmentation to create more images overall for training.  

* Design, train and test a model architecture  


* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


