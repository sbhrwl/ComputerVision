# Networks

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
