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

browseURL('https://github.com/chrishamblin7/Subitizing_Networks')

#This code consists in a dataset generator, and the training scripts for a CNN:
#Data Set Generator:
  #generates 12,000 100x100 images with blue and red circular dots.
  #minimum number of dots in image:3
  #maximum number of dots in image:40
  #minimum dot ratio: 19:21
  #maximum dot ratio: 2:1
  #All images are controlled for simple image statistics, like the area cover by each color

  #Here's a sample image:
    browseURL('https://github.com/chrishamblin7/Subitizing_Networks')    #change this to image









# The responses of the network on a 2000 image testing set were written to this csv over the course of 80 epochs

cnn_data <- read.csv("simple_dot_comparison.csv")

#Lets add some columns by pulling from the 'img_name' which has latent info about the content of the images
num_blue <- vector() #number of blue dots in image
num_red <- vector() # number of red dots in image
num_total <- vector() #number of total dots in image
ratio_factor <- vector() #actual ratio of numbers coded as a factor
ratio_numeric <- vector() #fractional value of ratio (bigger/smaller)
bigger <- vector() #which color has more dots
correct <- vector() # was the neural network correct in its response of which color had more dots?

for (i in 1:nrow(simplecnn_data)) {
  if (cnn_data$prediction[i] == 0) {
    cnn_data$prediction[i] <- 'blue'
  }
  else {
    cnn_data$prediction[i] <- 'red'
  }
  if (cnn_data$prediction[i] ==  cnn_data$target[i]) {
    correct[i] <- 1
  }
  else {
    correct[i] <- 0
  }
  matches <- regmatches(as.character(cnn_data$img_name[i]), gregexpr("[[:digit:]]+", as.character(cnn_data$img_name[i])))
  numbers <- as.numeric(unlist(matches))
  num_total[i] <- numbers[1]
  num_blue[i] <- numbers[2]
  num_red[i] <- numbers[3]
  ratio_factor[i] <- paste(as.character(numbers[2]),as.character(numbers[3]),sep=":")
  if (numbers[2] > numbers[3]) {
    bigger[i] <- 'blue'
    ratio_numeric[i] <- numbers[2]/numbers[3]
  }
  else {
    bigger[i] <- 'red'
    ratio_numeric[i] <- numbers[3]/numbers[2]
  }
}

cnn_data$correct <- correct
cnn_data$num_blue <- num_blue
cnn_data$num_red <- num_red
cnn_data$num_total <- num_total
cnn_data$bigger <- bigger
cnn_data$ratio_factor <- ratio_factor
cnn_data$ratio_numeric <- ratio_numeric

