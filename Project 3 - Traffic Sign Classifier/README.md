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

The histogram above shows an overlay of each data set across each of the 43 different sign classes. Of additional note is that there are limited images available for individual classes, some with as little as 180 total images, which may make those images harder to classify accurately. I will want to use data augmentation to create more images overall for training.    
  
Number of training examples = 34,799  
Number of validation examples = 4,410  
Number of testing examples = 12,630    

To add more training images from the existing training data, I used a rotation data augmentation technique, whereby taking each image in the training data set and rotating it both 10 degrees from vertical to 350 degrees (-10 degrees) from vertical.  This brought the new amount of training examples available to train against to 104,397.  
  
![Figure_3](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%203%20-%20Traffic%20Sign%20Classifier/project_screenshots/rotated.jpg)  


* Design, train and test a model architecture  

Prior to training the model, I normalized all the training, validation and test data. The image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 127.5)/ 255` will take the pixel is a quick way to approximately normalize the data and can be used in this project. A well conditioned image allows the optimizer to be more efficient to find a solution.

Secondly, I converted all of the images to `grayscale` to help with image classification.  As Pierre Sermanet and Yann LeCun mentioned in their [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), using color channels didn't seem to improve things a lot.  Therefore, I will only use a single channel in my model, e.g. grayscale images instead of color RGB.  

My model architecture (Deep Learning Model) Based on LeNet Architecture, with the exception of adding max pooling layers, ReLU's sub sampling methods and dropout regularizations.  

![LeNet Architecture](https://github.com/silverwhere/Self-Driving-Car-Nanodegree---Udacity/blob/main/Project%203%20-%20Traffic%20Sign%20Classifier/lenet.png)
Source: Yan LeCun

#### Parameters  

To calculate the number of neurons in each layer on our CNN:

Given: 
*  Input Layer has a Width `W` and a Height `H`  
*  Convolution Layer has a Filter size `F`
*  Stride of `S`
*  Padding `P`  Note no padding was neccessary for image library (images are 32x32x3)
*  Number of Filters `K`

Formula:  
*  Width of Next Layer `W_out = [(W - F + 2P) / S] + 1`
*  Height of Next Layer `H_out = [(H - F + 2P) / S] + 1`
*  Output Depth `D_out = K` Number of filters
*  Output Volume `V_out = W_out * H_out * D_out`  

With parameter sharing, each neuron in an output channel shares it's weights with every other neuron in that channel.  So the number of parameters is equal to the number of neurons in the filter, plus a bias neuron, all multiplied by the number of channels in the output layer.

**Remember with weight sharing we use the same filter for an entire depth slice!**

If you have N inputs, & K outputs you have (N+1)K parameters to use.  
  
* Use the model to make predictions on new images  

Utilizing the model initially offered a great training accuracy, but a poor validation accuracy.  The training log in the `Traffic_Sign_Classifier.ipynb` file will list the changes.  But most notable were:  
* data augmentation training images (rotated) preprocessing at this step  
* shuffled training data
* normalization (changed normalization to -1 to 1 from 0 to 1)
* LeNet model architecture modification - original LeNet model plus added I a dropout before final 3rd fully connected layer.  The course mentioned an initial keep_prob: 0.7 for a dropout regularization, but from From Nitish Srivastava [paper](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) *"Dropout probability p independent of other units, where p can be chosen using a validation set or can simply be set at 0.5, which seems to be close to optimal for a wide range of networks and tasks."  Therefore, I choose a keep_prob: 0.5.    

Overall my results were great!  

**Training Accuracy = 99.6%     
Validation Accuracy = 96.1%    
Test Set Accuracy = 93.1%**  

* Analyze the softmax probabilities of the new images



