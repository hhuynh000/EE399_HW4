# Feed Foward Neural Network for Hand Digits Classification
Huy Huynh

## Abstract
This page explores the use of feed foward neural network for regression and classification tasks. Draw comparision between neural networks approach and traditional machine learning approaches like Linear Discriminate Anaylsis (LDA), Support Vector Machines (SVM) and Decision Tree.

## Introduction
Firstly, test the performance of feed foward neural network on linear regression and compare with least square approach on a simple function $f(x) = A\cos{(Bx)}+Cx+D$. Then test the performance for classication on MNIST data set and compared with LDA, SVM and Decision Tree.

## Background
Neural network are essentially linear mapping between each layers, but with non-linearality added through the use of activation function. The weights are solved using back propagation, which make use of chain rule to compute the gradient in combination with a simple activation function that is easily differentialable. The activation function used is ReLU, shown in figure 2.

<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW4/blob/main/resources/nn.png" width="500"/>
</p>
<p align="center">
  Figure 1. Neural Networks Example Diagram
</p>

<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW4/blob/main/resources/relu.png" width="500">
</p>
<p align="center">
  Figure 2. ReLU Function
</p>

## Implementation
The neural network architecture for the regression is 3 layers that take 1 dimensional input x and expand to 2 layers with 64 nodes and output a 1 dimensional vector y. The loss function used is mean squared error and optimizer used is Adam with a learning rate of 0.0005 and 100 epochs of training. The nerual network code is shown in the figure below.

The data points used in regression task is shown in the figure below.
<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW4/blob/main/resources/reg_nn.png" width="500">
</p>
<p align="center">
  Figure 3. Function $f(x) = A\cos{(Bx)}+Cx+D$ noisy data points
</p>

The neural network architecture for the classification is 3 layers that take a flat 748 dimensional input (image) that goes through a layer with 128 nodes, then another layer of 64 nodes then output a 10 dimensional vector representing the probabilities guess for al 10 digits. The neural network classification will then be the index corresponding to largest value in the output vector. The loss function used is cross entropy and optimizer used is Adam with a learning rate of 0.0005 and 30 epochs of training. The nerual network code is shown in the figure below.

The data points used in regression task is shown in the figure below.
<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW4/blob/main/resources/reg_data.png" width="500">
</p>
<p align="center">
  Figure 4. Function $f(x) = A\cos{(Bx)}+Cx+D$ noisy data points
</p>

## Result
### Regression
The data points used in regression task is shown in the figure below.
<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW4/blob/main/resources/reg_data.png" width="500">
</p>
<p align="center">
  Figure 5. Function $f(x) = A\cos{(Bx)}+Cx+D$ noisy data points
</p>

When the first 20 data points is used for training and the remaining 10 data points is for testing, the training mse is 86.58 and the testing mse is 269.68. The resulting prediction from the neural network is shown in the figure below.
<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW4/blob/main/resources/end.png" width="500">
</p>
<p align="center">
  Figure 6. Neural Network Regression
</p>

When the first 10 and last 10 data points is used for training and the remaining middle 10 data points is for testing, the training mse is 41.61 and the testing mse is 91.84. The resulting prediction from the neural network is shown in the figure below.
<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW4/blob/main/resources/middle.png" width="500">
</p>
<p align="center">
  Figure 7. Neural Network Regression
</p>

In comparision to least square regression, the prediction from the neural network most closely similar to fitting the data with a linear model using least square. The resulting prediction from the neural network seem to try to fit a linear line to the data. Similar to least square regression, using the first and last 10 data points significantly increase the model ability to predict the overall data points.

### Classification
The MNIST data set are 28x28 pixel images of handwritten digits. Some example of the images are shown in the figure below.
<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW4/blob/main/resources/mnist.png" width="500">
</p>
<p align="center">
  Figure 8. MNIST Images
</p>

Performing PCA to get the first 20 principal components to show potential feature space of the data. The first 20 principal components of the MNIST data set is shown in the figure below.
<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW4/blob/main/resources/pca.png" width="500">
</p>
<p align="center">
  Figure 9. MNIST PCA
</p>

After training the classification neural network described above, the model accuracy on the testing data is 97%.
