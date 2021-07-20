# Computer Vision

- [Drawbacks of ANN](#drawbacks-of-ann)
- [Convolution Neural Network](#convolution-neural-network)
  - [Principals](#principals)
  - [Kernel](#kernel)
  - [CNN in Action](#cnn-in-action)
    - [Steps in Convolution operation](#steps-in-convolution-operation)
    - [Convolution done on RGB channels with 3 * 3 kernel](#convolution-done-on-rgb-channels)
    - [Max Pooling](#max-pooling)
    - [CNN Example](#cnn-example)
- [Features extracted at different layers](#features-extracted-at-different-layers)
- [Local and Global Receptive field](#local-and-global-receptive-field)
- [Dropout](#dropout)
- [Data Augmentation](#data-augmentation)
- [Hyper parameter Tuning](#hyper-parameter-tuning)
- [Experiments](#experiments)
- [BP in CNN](#bp-in-cnn)
- [Networks](#networks)
  - [LeNet](#lenet)
  - [AlexNet](#alexnet)
  - [VGG net](#vgg-net)
- [References](#references)

## Drawbacks of ANN
- Loss of features
  - Before feeding an image to the hidden layers of an MLP, we must flatten the image matrix to a 1D vector, this implies that all the image's 2D information is discarded.
- Spatial Invariance 
  - The information regarding spatial relation between different parts of objects in a 2D image are lost when it is flattened to a 1D vector input.
- Increase in number of Parameters to train
  - Consider  an image with dimensions 1000 × 1000, it will yield 1 million parameters for each node in the first hidden layer.
    So if the first hidden layer has 1,000 neurons, this will yield **1 billion parameters** even in such a small network. You can imagine the computational complexity of optimizing 1 billion parameters after only the first layer

# Convolution Neural Network
## Principals
- **Translation Invariance**
  In the earliest layers, our network should **respond similarly to the same patch**, regardless of where it appears in the image. This principle is called translation invariance.
- **Locality Principle**
  The earliest layers of the network should **focus on local regions, without regard for the contents of the image in distant regions**. This is the locality principle.
  Eventually, these local representations can be aggregated to make predictions at the whole image level.
  
## Kernel
- Kernel is also referred as Filter or Feature Extractor

## [CNN in Action](https://colab.research.google.com/drive/1xZcozBfAjRvmXWpX7F2IwZtrtV5utIWQ?usp=sharing)
### Steps in Convolution operation
- Matrix Calculation
- Padding
  - ZERO Padding
  - Reflective Padding
- Stride
- Feature accumulation
- Feature aggregation

### Convolution done on RGB channels
#### Example with 3 * 3 kernel
  * Same Filter/Kernel convolve 3 channels
  * Finally, after feature aggregation we have 1 extracted feature
  * Similarly, after using more filters, we would **extract** more features from the image
  * 1 Filter = 1 Feature
    * So, if we use **10 Filters** on our image, we will have **10 Features**
  * **Feature Map**: Container or Bucket that holds **all the extracted features** from all the channels as an output of convolution operation
    
![](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581a58be_convolution-schematic/convolution-schematic.gif)

<centre>Convolution with 3×3 Filter. [Source](http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution)</centre>

### Max Pooling
<img src="https://github.com/sbhrwl/ComputerVision/blob/main/artifacts/images/maxpool.jpeg"/>

Max Pooling with 2x2 filter and a stride of 2

### CNN Example
![CNN Image](https://res.cloudinary.com/practicaldev/image/fetch/s--w1RZuJPn--/c_imagga_scale,f_auto,fl_progressive,h_420,q_auto,w_1000/https://dev-to-uploads.s3.amazonaws.com/i/1inc9c00m35q12lidqde.png)

* Input Image has dimension 28 * 28
* Convolution parameters
  * Kernel 3 * 3
  * Padding of 1
  * Stride of 1
  * Activation function **Relu**
* 32 Filters applied to image at layer 1 => Feature map with 32 extracted Features
  * Max pooling reduced size from 28 to 14
* 32 Filters applied to image at layer 2 => Feature map with 32 extracted Features
* 64 Filters applied to image at layer 3 => Feature map with 64 extracted Features
  * Max pooling reduced size from 14 to 7
* 64 Filters applied to image at layer 4 => Feature map with 64 extracted Features

## Features extracted at different layers
* Edges
* Textures
* Pattern
* Parts of an object
* Objects in Image
* Image

<img src="https://github.com/sbhrwl/ComputerVision/blob/main/artifacts/images/filters.png" width="800"/>

[notebook](https://colab.research.google.com/drive/1tNle1niW_5rDf9wUL_HNY6rG1l54rGIA?usp=sharing)

- **Convolution Layer** is also referred as **Convolution block**
- **Max pooling Layer** is also referred as **Transition block**

## Local and Global Receptive field

## Dropout
Dropout layer makes Pixels black (CNN)
### Where to place
- Use drop out after flattening layer and in between FCs (mostly used)
- Use drop out before output layer (softmax/sigmoid)

### What not to do
- Do not use dropout between Convolution layer
- Do not use dropout at First Convolution block (we are capturing features)
  and Last Convolution block (we have all extracted features we would not want to loose them)
- Do not use Convolution layer in transition block (MP layer)

## Data Augmentation
- You can also use a data augmentation type (augmentation_type) hyper parameter to configure your input images to be augmented in multiple ways. 
- You can randomly 
  - Crop the image and 
  - Flip the image horizontally and 
  - Alter the color using the Hue-Saturation-Lightness channels.
  - Patching

Augmentation Types
- Run time
- Pre processed

## Hyper parameter Tuning
- Image Resolution
- Kernel Size: 3 * 3, 5 * 5, 7 * 7
- Stride: 2, 3
- Padding: same, valid
- Activation Function: Relu
- Dense Layer: 64, 128
- Loss Function: Categorical sparse entropy
- Optimizers: Adam, SGD
- Number of Convolution layers
- Image reduction (Max Pool layers)
- Epochs
- Steps per epoch = Total number of samples/Batch size
  - Default batch size in keras is 32
  - We have 2000 sample images
  - Steps per epoch = 2000/32 = 62

## Note
- Batch Normalisation
- BP and FP in terms of CNN
- NO back propagation for Max pooling and FC layers
- BP starts from Entry side of the model (first layer)
- TF1.14 is stable but old


## Experiments
### Architecture 1
| Layers              | Kernel Size | Stride | Padding | Number of Kernels |Activation  |
| ------------------- | ----------- | -------| --------| ----------------- |----------- |
| Convolution Layer 1 | 3 * 3       | 1      | 1       | 32                | Relu       |
| Max Pooling         | 2 * 2       | 2      | 2       |                   | Relu       |
| Convolution Layer 2 | 3 * 3       | 1      | 1       | 32                | Relu       |
| Max Pooling         | 2 * 2       | 2      | 2       |                   | Relu       |
| Convolution Layer 3 | 3 * 3       | 1      | 1       | 64                | Relu       |
| Max Pooling         | 2 * 2       | 2      | 2       |                   | Relu       |
| Flatten             |             |        |         |                   |            |
| Dense/FC            |             |        |         |                   |            |
| Dropout             |             |        |         |                   |            |
| Dense/FC            |             |        |         |                   |            |
| Dropout             |             |        |         |                   |            |
| Output layer        |             |        |         |                   | Softmax    |

## [BP in CNN](https://github.com/sbhrwl/ComputerVision/blob/main/CNN_BP.md)

## Networks
### [LeNet](https://drive.google.com/file/d/1I1aqvBPQ0qxOSsLr5wdNe-LNQHySmgBS/view?usp=sharing)

### [AlexNet](https://drive.google.com/file/d/1Sc5wUxu5hlAzarK91vWL2So9ECuGRPPy/view?usp=sharing)

### [VGG net](https://drive.google.com/file/d/18PQ867fQRULK-mD7gQkEOJv5_Cv2pv5v/view?usp=sharing)

## References
- Kernels
  - [Kernels](https://setosa.io/ev/image-kernels/)
  - [Kernels as Edge Detector](https://aishack.in/tutorials/image-convolution-examples/)
- Convolution Visualisation
  - [2D visulaisation](https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html)
  - [Tensor playground](https://tensorspace.org/html/playground/lenet.html)
  - [CNN explainer](https://poloclub.github.io/cnn-explainer/)
  - [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/)
  - [NB](https://jovian.ai/paulbindass/convolutional-neural-network-world)
- Data Augmentation
  - [Library imgaug](https://github.com/aleju/imgaug)
  - [Augmentor](https://augmentor.readthedocs.io/en/master/)
  - [Augmentor github](https://github.com/mdbloice/Augmentor)
  - [Research paper- Cutout](https://arxiv.org/pdf/1708.04552.pdf)
  - [Research paper- Image Cropping and Patching](https://arxiv.org/pdf/1811.09030.pdf)
  - [Library Albumentations](https://github.com/albumentations-team/albumentations)
  
  
