import csv # csv libary
import cv2 
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy import ndimage

# Global Parameters
epochs = 5
batch_size = 32
validation_split = 0.2
correction = 0.2


# Read in each row/line from driving_log.csv
lines = [] # samples
with open('training_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
'''
Generators can be a great way to work with large amounts of data. Instead of storing the preprocessed data in memory all at once, using a generator you can pull pieces 
of the data and process them on the fly only when you need them, which is much more memory-efficient.

A generator is like a coroutine, a process that can run separately from another main routine, which makes it a useful Python function. Instead of using return, the
generator uses yield, which still returns the desired output values but saves the current values of all the generator's variables. When the generator is called a second
time it re-starts right after the yield statement, with all its variables set to the same values as before.
'''        
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path ='training_data/IMG/' + filename
                    image = ndimage.imread(current_path)
                    images.append(image)

                # create adjusted steering measurements for the side camera images
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # add angles to data set
                measurements.extend([steering_center])
                measurements.extend([steering_left])
                measurements.extend([steering_right])
            
            # Data augmentation
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement)
                augmented_measurements.append(measurement*-1.0)

            # Keras requires arrays. Convert images and steering measurements to numpy arrays
            X_train = np.array(augmented_images) # features from images
            y_train = np.array(augmented_measurements) # ground truth measurments
            
            # shuffle the data
            yield shuffle(X_train, y_train) 

# Utilize Generators
train_samples, validation_samples = train_test_split(lines, test_size=validation_split)
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Setup Keras
from keras.models import Sequential # The keras.models.Sequential class is a wrapper for the neural network model. It provides common functions like fit(), evaluate(), and compile().
from keras.models import Model # Create the Sequential model
from keras.layers import Lambda 
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D  #
from keras.layers.pooling import MaxPooling2D

# Build the Neural Network Architecture
model = Sequential()
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(160,320,3))) # normalize the image by 255 the maximum for an image pixel between 0 and 1, then mean-center the image by subtracting -0.5 from each element, and give a image pixel of -0.5 to 0.5

model.add(Cropping2D(cropping=((75,25),(0,0)))) # cropping image 75 pixels from the top and 25 from the bottom, from "Even more powerful network video"

#NVIDIA END TO END NETWORK WITH MAX POOLING AND DROPOUTS ADDED AS DENOTED
model.add(Conv2D(24, (5,5), padding='valid', activation='relu')) # 24 filters 5x5 kernal
model.add(MaxPooling2D()) #ADDED
model.add(Dropout(0.5)) # ADDED dropout rate set to 0.5 for training/validation
model.add(Conv2D(36, (5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D()) #ADDED
model.add(Conv2D(48, (5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D()) #ADDED
model.add(Conv2D(64, (3,3), padding='valid', activation='relu'))
model.add(Conv2D(64, (1,1), padding='valid', activation='relu'))
model.add(MaxPooling2D()) #ADDED
model.add(Dropout(0.5)) # dropout rate set to 0.5 for training/validation
model.add(Flatten())

# Next, four fully connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1)) # single output node = predicted steering angle

model.summary() # print a model summary

# To compile the keras model use a "mean squared error" loss function, good for regression and "Adam" optimizer
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, steps_per_epoch = ceil(len(train_samples)/batch_size), validation_data = validation_generator, validation_steps = ceil(len(validation_samples)/batch_size), epochs=epochs, verbose=1)

# save trained model architecture
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### Keras outputs a history object that contains the training and validation loss for each epoch.
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')

# save the file (see next line)
plt.savefig('examples/mean_squared_error_loss.png')





