# Networks
  - [LeNet](#lenet)
  - [AlexNet](#alexnet)
  - [VGG net](#vgg-net)
  - [Inception](#inception)
    - [Intution](#intution)
    - [Core Idea](#core-idea)
    - [Naive Version](#naive-version)
    - [Dimension Reduction version](#dimension-reduction-version)
    - [Auxillary Classifier](#auxillary-classifier)
    - [Global Average Pooling](#global-average-pooling)
    - [Inception Network Design](#inception-network-design)
  - [InceptionV2](#inceptionv2)
  - [InceptionV3](#inceptionv3)
  - [ResNet](#resnet)
  - [Model available in Keras](#model-available-in-keras)

## [LeNet](https://colab.research.google.com/drive/1N29L05F972fB97JL0mdqQwc3os2OB-Me?usp=sharing)
* LeNet-5 is a very efficient convolutional neural network for **handwritten character recognition**
* LeNet convolutional neural networks can make good use of the structural information of images.
* In LeNet convolutional layers had fewer parameters

## [AlexNet](https://colab.research.google.com/drive/1RvvNKpxc0z2xtunAe13un1Ri_SzITTMG?usp=sharing)
### Why does AlexNet achieve better results?
1. Relu activation function was used
2. Standardization ( Local Response Normalization ) was performed
3. Dropout was used
4. Enhanced Data via Data Augmentation

## [VGG net](https://colab.research.google.com/drive/1wNoiB5e8iHFKH0YXp7H6iOTG8Y2a5RL2?usp=sharing)
<img src='https://github.com/sbhrwl/ComputerVision/blob/main/artifacts/images/vgg_network.png'>

[Visualise model](https://ethereon.github.io/netscope/#/preset/vgg-16)

### Summary of VGGNet improvement points
1. A smaller 3 * 3 convolution kernel and a deeper network are used . 
2. The stack of two 3 * 3 convolution kernels is relative to the field of view of a 5 * 5 convolution kernel, and the stack of three 3 * 3 convolution kernels is equivalent to the field of view of a 7 * 7 convolution kernel. 
3. In this way, there can be fewer parameters (3 stacked 3 * 3 structures have only 7 * 7 structural parameters (3 * 3 * 3) / (7 * 7) = 55%); on the other hand, they have more The non-linear transformation increases the ability of CNN to learn features.
4. In the convolutional structure of VGGNet, a 1 * 1 convolution kernel is introduced. 
5. Without affecting the input and output dimensions, non-linear transformation is introduced to increase the expressive power of the network and reduce the amount of calculation.
6. During training, first train a simple (low-level) VGGNet A-level network, and then use the weights of the A network to initialize the complex models that follow to speed up the convergence of training .

**What is the role of 1x1 convolution kernel**
- 1x1 kernel **increases the nonlinearity of the model** without affecting the receptive field
- 1x1 winding machine is equivalent to linear transformation, and the non-linear activation function plays a non-linear role

## Inception
### Intution
#### Drawbacks of Fully connected/Dense layers
- Trainable parameters increases exponentially
- Longer training time
- Prone to Overfitting
#### Unconventional Convolution Neural Network
- Fully connected layers replaced with Convolution 
- [Paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf)
- [Notebook](https://colab.research.google.com/drive/1diTcRiepAFMSUCG-62hJYInn4wYPVK-i?usp=sharing)
- Notice Convolution layer instead of FC layers when closing the network
    ```
    # Performing 2dconvolution followed by BatchNormalization and Dropout
    model.add(Convolution2D(16, 3, 3, activation='relu'))#4                      # channel dimensions = 4x4x16    and Receptive field = 22x22
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    # Performing only 2dconvolution at the last convolution layer(no batchnormalization and dropout)
    model.add(Convolution2D(10, 4, 4))                                           # using 4x4 kernel to see the complete image

    # Here we are Flateening our dat i.e making it one dimensional which we will feed to the network.
    model.add(Flatten())
    #Using softmax activation function at the last layer which is used for multi class classification
    model.add(Activation('softmax'))
    ```
#### Core Idea
- Increase the **depth and width of the network** by continuously copying the inception modules

### Inception Module: Network in Network
**Four parallel channels:**
* 1x1 conv: Reduces the dimesnsion of **Input feature map** without too much loss of the input spatial information.
* 1x1conv followed by 3x3 conv: 3x3 conv **increases the receptive field of the feature map**, and changes the dimension through 1x1 conv
* 1x1 conv followed by 5x5 conv: 5x5 conv **further increases the receptive field of the feature map**, and changes the dimensions through 1x1 conv
* 3x3 max pooling followed by 1x1 conv: 
  * The author believes that although the pooling layer will lose space information, it has been effectively applied in many fields, which proves its effectiveness, 
  * So a parallel channel is added, and it is changed by 1x1 conv for its output dimension.

<img src='https://github.com/sbhrwl/ComputerVision/blob/main/artifacts/images/inception.png'>

### Naive Version
- Consider an Input from previous layer as **28x28x196**
- Lets assume that we are using **N** Filters/Kernels/Features with our next stage of convolutions
- Output of 1x1 will be **28x28x(N)**
- Output of 3x3 will be **26x26x(N)**
- Output of 5x5 will be **24x24x(N)**
- Output of Max Pooling will be **14x14x(N)**

### Dimension Reduction version
- As 1x1 reduces the number of channels in the output
- Using 1x1 enables us to have a **Security Border**
- We can use different 1x1 for each convolution, ex
  - 32 channels for 1st 1x1
  - 64 channels for 3x3 conv
  - 128 channels for 5x5 conv
  - 96 conv for 1x1 after Max pooling
- Reducing the number of channels makes our model **Computationally less expensive**
- As after each convolution the shape has been reduced (as reduced number of channels with 1x1), we then use **Padding** to get the same dimension in Output as was with the previous layer
- Now Concatenation can be performed (Filter concatenation) at next module
- **With independent 1x conv (1st block) we always have One copy of image from previous layer**

### Auxillary Classifier
- Auxillary Classifier serves as an **Exit point** for our network
  - Feature map from **Inception block** is sent to **Average pool** 
  - Followed by 1x1 con to reduce number of channel
  - 2 FCs
  - Softmax activation for classification
- Auxillary Classifier helps for Feature Visualisation
- Auxillary Classifier also caters to **Vanishing Gradient** problem as well because it serves as closing the networks, so smaller network to Back Propogate hence vanishing gradient is handled

### [Activation Map](https://towardsdatascience.com/live-visualisations-of-cnns-activation-maps-using-tensorflow-js-27353cfffb43)
- Activation Map is output of particular convolution layer. we can use activation maps for visualisation of CNN.

#### Parameter Comparison with GAP and FC layers
<img src="https://github.com/sbhrwl/ComputerVision/blob/main/artifacts/images/GAP-parameterComparison.png" width=500>

### Global Average Pooling
- One advantage of global average pooling over the fully connected layers is that it is **more native to the convolution structure by enforcing correspondences between feature maps and categories**. 
  - Thus the feature maps can be easily interpreted as categories confidence maps. 
- Another advantage is that there is **no parameter to optimize** in the global average pooling thus overfitting is avoided at this layer. 
- Furthermore, global average pooling sums out the spatial information, thus it is more robust to spatial translations of the input.

#### [Inception Network Design](https://colab.research.google.com/drive/1P-0FBj2f0KEsgNevk2SBNzdkPIhnIIrT?usp=sharing)
**Note**: 
- Inception-v1 was introduced in 2014
- It is also called as GoogleNet

## InceptionV2
- Inception-v2 (2015) uses **Batch Normalization**
- Batch Normalization accelerates the training of the networkand also reduces the degree of overfitting of the network. 

### Algorithm advantages:
1. **Improved learning rate** : 
    - In the BN model, a **higher learning rate** is used to accelerate training convergence, but it will not cause other effects. 
    - Usually, the minimum learning is required to ensure the loss function to decrease, 
    - But the BN layer keeps the scale of each layer and dimension consistent, so you can directly use a higher learning rate for optimization.
2. **Remove the dropout layer** : 
    - The BN layer makes full use of the goals of the dropout layer. 
    - Remove the dropout layer from the BN-Inception model, but no overfitting will occur.
3. **Decrease the attenuation coefficient of L2 weight** : 
    - Although the L2 loss controls the overfitting of the Inception model, **the loss of weight has been reduced by five times in the BN-Inception model**
4. **Accelerate the decay of the learning rate** : 
    - When training the Inception model, we let the learning rate decrease exponentially. 
    - Because our network is faster than Inception, we will **increase the speed of reducing the learning rate by 6 times**.
5. **Remove the local response layer** : 
    - Although this layer has a certain role, but after the BN layer is added, this layer is not necessary.
6. **Scramble training samples more thoroughly** : 
    - We scramble training samples, which can prevent the same samples from appearing in a mini-batch. 
    - This improves the accuracy of the validation set by 1%, which is the advantage of the BN layer as a regular term. 
    - In our method, random selection is more effective when the model sees different samples each time.
7. **To reduce image distortion**: 
    - Because BN network training is faster and observes each training sample less often, we want the model to see a more realistic image instead of a distorted image.

[notebook](https://colab.research.google.com/drive/1bbhZYdvBueu2OH7i-GaOFD-TtVHonBLK?usp=sharing)

## InceptionV3
- **Label-smoothing**
- **BN-auxiliary**
- The computational cost is reduced while improving the accuracy of the network

### General Design Principles
1.  **Prevent bottlenecks in characterization**
    - 1x1 conv in inception block (reduce before 3*3 and 5*5) are termed as bottlenecks as they reduce the number of channels/feature
    - 1*1 or Pointwise layer or Bottleneck layers
    - The so-called bottleneck of feature description is that a large proportion of features are compressed in the middle layer (such as using a pooling operation). 
    - This operation will cause the loss of feature space information and the loss of features. (Later Hole Convolution operations)
2.  **The higher the dimensionality of the feature, the faster the training converges** . 
    - That is, the independence of features has a great relationship with the speed of model convergence.
    - The more independent features, the more thoroughly the input feature information is decomposed. 
    - It is easier to converge if the correlation is strong. 
    - **Hebbin principle** : fire together, wire together.
3.  **Reduce the amount of calculation through dimensionality reduction**
    - In v1, the feature is first reduced by 1x1 convolutional dimensionality reduction. 
    - There is a certain correlation between different dimensions. 
    - Dimension reduction can be understood as a **lossless or low-loss compression**. 
      - Even if the dimensions are reduced, the correlation can still be used to restore its original information.
4.  **Balance the depth and width of the network**
    - Only by increasing the depth and width of the network in the same proportion can the performance of the model be maximized.

### Factorizing Convolutions with Large Filter Size
- With the same number of convolution kernels, larger convolution kernels (such as 5x5 or 7x7) are more expensive to calculate than 3x3 convolution kernels , which is about a multiple of 25/9 = 2.78

<img src="https://github.com/sbhrwl/ComputerVision/blob/main/artifacts/images/FactorizingConvolutions.png" width=500>

- 5x5 replaced with 2 **3x3 conv**

### Spatial Factorization into Asymmetric Convolutions

<img src="https://github.com/sbhrwl/ComputerVision/blob/main/artifacts/images/Spatial Factorization.jpg" width=600>

#### Benefits of Factorizing convolutions
- One convolution is broken into 2 lighter convolution making it 
  - Fit of weak GPUs
  - Faster training of network
  - Enables use to deploy our model in weak systems (poor hardware)

## ResNet
- ResNet, also known as residual neural network, refers to the idea of **​​adding residual learning** to the traditional convolutional neural network, 
- Residual learning solves the problem of 
  - Gradient dispersion and 
  - Accuracy degradation (training set) in deep networks, 

### Challenges faced by increasing depth
* The first problem brought by increasing depth is the problem of **Exploding gradient or Vanishing Gradient also referred as Gradient dissipation**
  - This is because as the number of layers increases, the gradient of backpropagation in the network will become **unstable with continuous multiplication**, and becomes either very high or negligibly smaller
  - This results in problem of Exploding gradient or Vanishing Gradient also referred as Gradient dissipation
  - Usually Vanishing Gradient is the case
* In order to overcome gradient dissipation, many solutions have been devised, such as 
  - Using BatchNorm, 
  - Replacing the activation function with ReLu, 
  - Using Xaiver initialization 
* Another problem of increasing depth is the problem of **network degradation**
  - As the depth increases, the performance of the network will become worse and worse, which is directly reflected in the decrease in accuracy on the training set. 
  - The residual network article solves this problem. 
  - And after this problem is solved, the depth of the network has increased by several orders of magnitude.

### Residual Block
- Convolution layer followed by Relu and BN
- [Paper](https://arxiv.org/pdf/1512.03385.pdf)
- [ResNet](https://www.youtube.com/watch?v=ZILIbUvp5lk)
- [Why ResNet works](https://www.youtube.com/watch?v=RYth6EbBUqM)
- A building block of a ResNet is called a residual block or identity block. 
- A residual block is simply when the activation of a layer is fast-forwarded to a deeper layer in the neural network
<img src="https://github.com/sbhrwl/ComputerVision/blob/main/artifacts/images/ResidualBlock.png" width=400>

- This is also termed as **Skip Connection**
  - Advantages of Skip connection
    - Output from Residual block is concatenated with Input
    - Amplification of Gradients (amplification of gradients overcomes vanishing gradient problem)
    - During BP, we do not update weights of skipped layer
      - Advantage: Lesser parameters to train
  - Skip is not happening everytime, in keras implementation, skip connection takes place for every BN layer
    - [keras implementation](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py)
    ```
    if conv_shortcut is True:
    shortcut = layers.Conv2D((64 // groups) * filters, 1, strides=stride,
                             use_bias=False, name=name + '_0_conv')(x)
    shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                         name=name + '_0_bn')(shortcut)
    ```
  - Pathways for Gradient Traversal
  <img src="https://github.com/sbhrwl/ComputerVision/blob/main/artifacts/images/ResnetGradientTraversal.jpg" width=600>
   
    - Pathway 1 is with Skip connection
    - Pathway 2 is normal FP

#### How are 1x1 used in ResNet
- For deeper networks, 
  - we all use the bottleneck structure on the right side of below figure
  - first using a 1x1 convolution for dimensionality reduction, 
  - then 3x3 convolution, and 
  - finally using 1x1 dimensionality to restore the original dimension.
- How is **cost of the calculation** of the residual block is optimized?
  - The two 3x3 convolution layers are replaced with 1x1 + 3x3 + 1x1 , as shown below. 
  - The middle 3x3 convolutional layer in the new structure first reduces the calculation under one dimensionality-reduced 1x1 convolutional layer , and 
  - then restores it under another 1x1 convolutional layer , both **maintaining accuracy and reducing the amount of calculation**.
<img src='https://github.com/sbhrwl/ComputerVision/blob/main/artifacts/images/ResnetCostOptimization.png'>

- 1x1 implementation in whole network
<img src="https://cdn-5f733ed3c1ac190fbc56ef88.closte.com/wp-content/uploads/2019/07/ResNet50_architecture-1.png">


- In ResNet, Residual Block (Conv+Relu+BN) is amplifying Gradients whereas in inception Auxillary block was amplifying the gradients
- When training network, check for supported default and minimum size of input image
  ```
  # Determine proper input shape
  input_shape = _obtain_input_shape(input_shape,
                                    default_size=224,
                                    min_size=32,
                                    data_format=backend.image_data_format(),
                                    require_flatten=include_top,
                                    weights=weights)
  ```
- Smaller resnet (<50) are not in keras



## [Model available in Keras](https://keras.io/api/applications/)

**From TF2 onwards we can add Padding with MP**
