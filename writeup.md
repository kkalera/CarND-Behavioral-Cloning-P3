# Behavioral cloning project

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build a cnn in Keras to predict steering angles from images
- Train and validate the model using training and validation sets
- Test the model driving autonomously around the track without leaving the road
- Summarize the results in a report

## Rubric Points

### Files submitted & Code quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My submission includes the followin files:

- car_training.py : Complete script to load the data, preprocess the data and train the model
- drive.py : Script for driving the car in autonomous mode
- model.h5 : The final trained model
- writeup.md : This writeup summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```python
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The car_trainingpy.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5 convolutional layers, followed by 4 fully connected layers.

The data is normalized in the model using a Keras lambda layer (code line 17).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (code lines 26 and 28).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 124-133). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code line 32).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and swerving from left to right

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As suggested by the instructors, I used the CNN architecture from Nvidia's paper. Since Nvidia had such good results with this, I didn't see the need to reinvent the wheel.

At first, the model would drive off the road on random spots, and after more training it went off road when encountering the dirt sideroad. This lead me to change my training data, about which I'll go into more detail below.

Although this was working great, the model was overfitting. This lead me to add 2 dropout layers to reduce overfitting and make the model generalize better.

Finally the model is able to drive around the track without going off road.

#### 2. Final Model Architecture

The final architecture can be found in the code from lines 16 - 30

#### 3. Creation of the Training Set & Training Process

This was one of the harder things to solve for me. I tried multiple approaches. I first started out with just center driving. This wasn't working as expected.

My second approach was adding data from driving from the side of the road back to center. Although this made improvements. It still wasn't what it needed to be. 

My final approach for data collection was inspired by Microsoft's Autonomous driving cookbook. They have a similar project to this one that can be found here: https://github.com/Microsoft/AutonomousDrivingCookbook/tree/master/AirSimE2EDeepLearning

In the exploration of the data, they show a set of data of center lane driving, aswell as a set of data of swerving from left to right across the road.

This ended up being the tactic I used, and this worked great. I discarded all the previous training data I had. Then did 4 laps of the track driving in the center, and then 1 lap swerving from the left to the right side. Doing this for both driving directions, gave me a dataset with 75% center lane driving and 25% swerving for riding in either direction.

