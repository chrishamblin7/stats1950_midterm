# PSY 1950 // Intermediate Statistical Analysis in Psychology
# Midterm // ANS neural networks
# October 17, 2019
# Chris Hamblin (chrishamblin@fas.harvard.edu)
# -----------------------------------------------------------------------


#Psychologists have long posited that people and animals have an innate approximate number system (ANS), 
#which allows them to get a rough sense for the number of items in a visual array at a glance. 
#A very reliable feature, perhaps the defining feature, of the ANS is that it obeys Weber's law.
#In the context of the ANS, Weber's law states that an indivual noticing the difference in the number of items
#in two arrays is a function of the ratio between the number of items in each array.
#The ratio at which people can just notice a difference, their Weber fraction, is constant, independent of the number of items.
#What's more, plotting ratio against discrimination accuracy should yeild an S-shaped curve, modelable with a sigmoid function:

sigmoid = function(x) {
  1 / (1 + exp(-x))
}

x <- seq(-5, 5, 0.01)
plot(x, sigmoid(x),)

#We can parameterize with 'a','b','xs' and 'ys' as below 
#to adjust the tightness of the S, its upper and lower bound and its shift along the x and y axes:

sigmoid = function(x,a,b,xs,ys) {
  b / (1 + exp(-a*(x-xs)))+ys
}

#We might expect performance on a dot discrimination task to look something like this:
plot(x, sigmoid(x,7,.5,1.5,.5), xlim = c(1,2.5), xlab = 'Dot Number Ratio', ylab='Accuracy', main='Dot Discrimination')

#This curve will vary depending on the age and species of the subject being tested.

#This analysis will probe whether the ANS is truly a designated system. Are these Weber-like properties 
#latent in the computations of the ANS itself, or rather are they a feature of the data the ANS operates over? 
#Will any arbitrary model tasked with distinguishing the greater of two sets show accuracies in accordance with Weber's law?
#For the midterm I'll test this with a simple, two layer convolutional neural network model.
#Most of the code written for this project is in python:

browseURL('https://github.com/chrishamblin7/stats1950_midterm/tree/master/python')

#This code consists in a dataset generator, and the training scripts for a CNN:

#Data Set Generator:

  #generates 12,000 100x100 images with blue and red circular dots.
  #minimum number of dots in image:3
  #maximum number of dots in image:40
  #minimum dot ratio: 19:21
  #maximum dot ratio: 2:1
  #All images are controlled for simple image statistics, like the area cover by each color

  #Here's a few sample images:
    browseURL('https://github.com/chrishamblin7/stats1950_midterm/tree/master/sample_dots')    #change this to image

#CNN
    
    #Layers:
      #convolution 1 : 3 channel --> 20 channel, 5x5 kernel, 1x1 padding
      #relu
      #2X2 max pooling
      #convolution 2 : 20 channel --> 50 channel, 5x5 kernel, 1x1 padding
      #relu
      #flatten to 22400 dimensional vector
      #fully connected linear transform to 500 dim vector
      #relu
      #fully connected linear transform to 2 dim vector
      #log of softmax output
    
    #Loss: negative log likelihood
    
    #Optimizer: Stochastic Gradient Descent
      #momentum: .5
      #Learning Rate: .01
    
    #Training:
      #100 size sample minibatches
      #10,000 image training set
      #2,000 image test set after each epoch
      #80 epochs
    
      
# The responses of the network on the testing set were written to this csv over the course of 80 epochs
cnn_data <- read.csv("simple_dot_comparison.csv")

#Lets add some columns by pulling from the 'img_name' which has latent info about the content of the images
epoch <- vector()
num_blue <- vector() #number of blue dots in image
num_red <- vector() # number of red dots in image
num_total <- vector() #number of total dots in image
ratio_factor <- vector() #actual ratio of numbers coded as a factor
ratio_numeric <- vector() #fractional value of ratio (bigger/smaller)
correct <- vector() # was the neural network correct in its response of which color had more dots?

# Let also set up a dataframe with aggregate info about each epoch, well populate it as we run through our for loop
# The columns will consist in accuracies associated with different ratio ranges 
epoch_data <- data.frame("epoch" = 1:81,)

for (i in 1:nrow(cnn_data)) {
  if (i%%1000 == 0) {
    print(i)
  }
  if (cnn_data$prediction[i] == 0) {     #change to meaningful label
    cnn_data$prediction[i] <- 'blue'
  }
  else {
    cnn_data$prediction[i] <- 'red'      #change to meaningful label
  }
  if (cnn_data$target[i] == 0) {         
    cnn_data$target[i] <- 'blue'
  }
  else {
    cnn_data$target[i] <- 'red'
  }
  if (cnn_data$prediction[i] ==  cnn_data$target[i]) {  
    correct[i] <- 1
  }
  else {
    correct[i] <- 0
  }
  epoch_matches <- regmatches(as.character(cnn_data$trial[i]), gregexpr("[[:digit:]]+", as.character(cnn_data$trial[i]))) #extract epoch info from text about trial
  epoch_numbers <- as.numeric(unlist(epoch_matches))
  epoch[i] <- epoch_numbers[1]
  img_matches <- regmatches(as.character(cnn_data$img_name[i]), gregexpr("[[:digit:]]+", as.character(cnn_data$img_name[i]))) #extract dot info from text about image
  img_numbers <- as.numeric(unlist(img_matches))
  num_total[i] <- img_numbers[1]
  num_blue[i] <- img_numbers[2]
  num_red[i] <- img_numbers[3]
  ratio_factor[i] <- paste(as.character(img_numbers[2]),as.character(img_numbers[3]),sep=":")
  if (img_numbers[2] > img_numbers[3]) {
    ratio_numeric[i] <- img_numbers[2]/img_numbers[3]
  }
  else {
    ratio_numeric[i] <- img_numbers[3]/img_numbers[2]
  }
}

#factorize factors
epoch <- factor(epoch)
ratio_factor <- factor(ratio_factor)
correct <- factor(correct)
cnn_data$prediction <- factor(cnn_data$prediction)
cnn_data$target <- factor(cnn_data$target)

#add to data table
cnn_data$epoch <- epoch
cnn_data$correct <- correct
cnn_data$num_blue <- num_blue
cnn_data$num_red <- num_red
cnn_data$num_total <- num_total
cnn_data$ratio_factor <- ratio_factor
cnn_data$ratio_numeric <- ratio_numeric

#This per item dataframe is kind of huge, lets make a smaller dataframe with aggregate statistics about each epoch



