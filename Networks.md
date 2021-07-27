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
    - [Inception Network Design](#inception-network-design)
  - [InceptionV2](#inceptionv2)
  - [InceptionV3](#inceptionv3)

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

### Another form of Factorized convolutions

<img src="https://github.com/sbhrwl/ComputerVision/blob/main/artifacts/images/FactorizedConvolutions-more.jpg" width=800>

#### Benefits of Factorizing convolutions
- One convolutin is broken into 2 lighter convolution making it 
  - Fit of weak GPUs
  - Faster training of network
  - Enables use to deploy our model in weak systems (poor hardware)

