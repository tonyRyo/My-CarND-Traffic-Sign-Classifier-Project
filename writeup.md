# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./predict/data_set_distribution.png
[image2]: ./examples/color.png "Color"
[image3]: ./examples/before_aug.png "Grayscale"
[image4]: ./examples/after_aug.png "rotate&trans"
[image5]: ./examples/1.png "Traffic Sign 1"
[image6]: ./examples/2.png "Traffic Sign 2"
[image7]: ./examples/3.png "Traffic Sign 3"
[image8]: ./examples/4.png "Traffic Sign 4"
[image9]: ./examples/5.png "Traffic Sign 5"
[image10]: ./predict/labels_accuracy.png "labels_accuracy"
[image11]: ./predict/1.png "predict Traffic Sign 1"
[image12]: ./predict/2.png "predict Traffic Sign 2"
[image13]: ./predict/3.png "predict Traffic Sign 3"
[image14]: ./predict/4.png "predict Traffic Sign 4"
[image15]: ./predict/5.png "predict Traffic Sign 5"
[image16]: ./examples/activationL2.png "activationL2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed.

![alt text][image1]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the shape,text,logo of the Traffic Sign are more meaningful which are not relate much to color data. Grayscale can also reduce the computation.  

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]![alt text][image3]


As a last step, I normalized the image data for data optimization.

I decided to generate additional data because  increasing the train set can train the model better without overfitting.

To add more data to the the data set, I used the following techniques to augment the data set with a factor of 4.
So the new data set size is 5 times to the orignal one.

Here is an example of an original image and an augmented image:

![alt text][image3]![alt text][image4]

The difference between the original data set and the augmented data set is the following,
1.Randomly translate the image with a constraint keeping the sign of the image inside the image window.
2.Randomly rotate the image with a rotate center of the traffic sign. The rotate angel is between -20~20 degree.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x16 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 13x13x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x16 	 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 4x4x16 	|
| RELU					|												|
| Fully connected		| 200 node. 									|
| Fully connected		| 84 node.  									|
| Fully connected		| 43 node.  									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer, a batch size of 128, epoch of 10, learning rate of 0.001.
I also use the dropout with a keep_prob of 0.7 to train the first and second fully connected layer. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.970
* validation set accuracy of 0.963
* test set accuracy of 0.951

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I chose LeNet for its good peformance at MINST.
* What were some problems with the initial architecture?
Not high enough accuracy rate even when I set the epoch to 50 times which mean it could bring about overfitting.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I broke down the two 5x5 convNet to three 3x3 to make the model deeper.What's more, I augment the data set five times to orignal one which could train the model better.
* Which parameters were tuned? How were they adjusted and why?
Epoch number. First I just increase the epoch from 10 to 50 to train the model but it dose not work well. Then I augment the data set and reset the epoch to 10.
Nodes number of the second fc layer.Since the input of the layer are increased from 400 to 512, I change the original node number of LeNet from 120 to 200.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I broke down the two 5x5 convNet to three 3x3 can make the model deeper to regconize more character of the traffic sign.
Convolution layer work well with this problem because the Three layer ConvNet can redeem the different characters of the traffic sign layer by layer.
And I use the dropout layer to prevent the overfitting of the model.

If a well known architecture was chosen:
* What architecture was chosen? 
Quasi simplified VGG convNet
* Why did you believe it would be relevant to the traffic sign application? 
Traffic sign is construct by line and features that the VGG network are good at. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Training, validation and test accuracy are 0.97,0.963,0.951 which are considerablely high to prove thar the model works well.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

All of the image are perfectly predicted.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep Ahead      		| Keep Ahead   									| 
| No entry     			| No entry 										|
| TurnLeft ahead		| TurnLeft ahead								|
| 60 km/h	      		| 60 km/h   					 				|
| Stop   	    		| Stop              							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the total accuracy on the test set of 0.951.

I also calculate the accuracy by each label using the test data-set, which could give an idea of the prediction reliability for the new images.

![alt text][image10]

Since accuracy of traffic sign label of the five image(which are 35,17,34,3,14) are higher than 0.9, the prediction of my model to the new five images is reliable.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th code-cell of the Ipython notebook shown as follow.

For the first image, the model is completely sure that this is a KeepAhead(35) sign, and the image does contain a KeepAhead sign. The top five soft max probabilities were

![alt text][image11]

For the second image, the model is completely sure that this is a NoEntry(17) sign, and the image does contain a NoEntry sign. The top five soft max probabilities were

![alt text][image12]

For the third image, the model is completely sure that this is a TurnleftAhead(34) sign, and the image does contain a TurnleftAhead sign. The top five soft max probabilities were

![alt text][image13]

For the fourth image, the model is completely sure that this is a 60kmh(3) sign, and the image does contain a 60kmh sign. The top five soft max probabilities were

![alt text][image14]

For the fifth image, the model is completely sure that this is a Stop(14) sign, and the image does contain a Stop sign. The top five soft max probabilities were

![alt text][image15]
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
![alt text][image16]
I tested the actvation L2 output by using a NoEntry sign image. It seems that the bold white line characteristic “-” and the circle  characteristic in the traffic sign are activated seperately.
