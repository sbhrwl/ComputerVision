## BP in CNN
### BP in Fully Connected Layers
#### Network for Classification
Flatter Layer -> Fully Connected layer -> Logits -> Softmax -> Output (Classification)

#### Forward Propagation for above network
- Output of FC layer is Logits
  - Logits never adds upto 1
- Output of Softmax is the Output classification
  - Softmax output adds upto 1
    
    <img src="https://render.githubusercontent.com/render/math?math=Softmax = \frac{e^{z_{i}}}{\sum_{i=1}^{k} e^{z_{k}}}">
    
    Logit output of Fully connected layer
    
    <img src="https://render.githubusercontent.com/render/math?math=\begin{vmatrix}z_{1}\\ z_{2}\\ z_{3}\\ z_{4}\end{vmatrix}">
    
    <img src="https://render.githubusercontent.com/render/math?math=\begin{vmatrix}0.23\\ 0.89\\ 0.45\\ 0.12\end{vmatrix}">

    This is now fed to Softmax, Output of Softmax 
    
    <img src="https://render.githubusercontent.com/render/math?math=\begin{vmatrix}y_{1}\\ y_{2}\\ y_{3}\\ y_{4}\end{vmatrix}">
    
    <img src="https://render.githubusercontent.com/render/math?math=\begin{vmatrix}0.05\\ 0.7\\ 0.2\\ 0.05\end{vmatrix}">
    
    **Output** class will be **y2**

#### Backward Propgation for above network
- Loss Function is cross entropy
  - For Binary classification
    
    <img src="https://render.githubusercontent.com/render/math?math=-(ylog(p)+(1-y)log(1-p))">

  - For MultiClass classification (classes>2)
    
    <img src="https://render.githubusercontent.com/render/math?math=-\sum_{c=1}^{M}y_{0,c}log(p_{0,c})">

  - During BP the Cross Entropy will be reduced based on new **weights** and **biases**
    
    - <img src="https://render.githubusercontent.com/render/math?math=w_{new} = w_{old} - \eta\frac{\partial y}{\partial w_{old}}">
    
    - <img src="https://render.githubusercontent.com/render/math?math=b_{new} = b_{old} - \eta\frac{\partial y}{\partial b_{old}}">

#### Lets say now, BP of Fully connected layers give us an output "gradient" <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L}{\partial w}">
This **gradient** will now be backpropogated to Maxx Pooling and Convolution layers

### BP in Max pooling and Convolution Layers
CNN L1 -> Max Pooling 1 -> ... -> Flatter Layer -> Fully Connected layer -> Logits -> Softmax -> Output (Classification)

#### [BP for Max pooling](https://www.youtube.com/watch?v=GH6qN0Bj8lA&t=1595s)
- Passing Gradients obtained from Fully connected layers (derivative of Loss function (for the output of Softmax function) wrt weights used at fully connected layers)
- Creating a Matrix of same size (as input image matrix)
- Filling the Matrix by passing the gradients as shown

<img src="https://github.com/sbhrwl/ComputerVision/blob/main/artifacts/images/max_pooling_bp.jpg">

- Now we have above Matrix of 4 * 4 gradients
- Lets call it as <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L}{\partial x^{`}}">
  - **x ~ w**
- These are newly calculated gradients based on gradients of fully connected layers
- These gradients will now be passed to Convolution layer present behind **this** Max pooling layer
- As parameters of Max pooling layer are **not trainable**, so there is **NO NEED to calculate Gradients** at Max pooling layer

#### BP for Convolution Layers
##### Intution
Lets say we have an Image 3 * 3 

<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}&space;0&space;&&space;0&space;&&space;0&space;\\&space;0&space;&&space;25&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;\end{bmatrix}" title="\begin{bmatrix} 0 & 0 & 0 \\ 0 & 25 & 0\\ 0 & 0 & 0 \end{bmatrix}" />

and the Filter 3 * 3

<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}&space;0&space;&&space;0&space;&&space;0&space;\\&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;\end{bmatrix}" title="\begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0\\ 0 & 0 & 0 \end{bmatrix}" />

The Convolution Operation would result in an Image of all **ZEROS**

<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}&space;0&space;&&space;0&space;&&space;0&space;\\&space;0&space;&&space;0&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;\end{bmatrix}" title="\begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0\\ 0 & 0 & 0 \end{bmatrix}" />

But Lets say, we introduce small variation in the filter and change filter with so that we have **1** in the center

<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}&space;0&space;&&space;0&space;&&space;0&space;\\&space;0&space;&&space;1&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;\end{bmatrix}" title="\begin{bmatrix} 0 & 0 & 0 \\ 0 & 1 & 0\\ 0 & 0 & 0 \end{bmatrix}" />

Now, the Convolution Operation would result in an Image with pixels value in center increased to **25**

<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}&space;0&space;&&space;0&space;&&space;0&space;\\&space;0&space;&&space;25&space;&&space;0\\&space;0&space;&&space;0&space;&&space;0&space;\end{bmatrix}" title="\begin{bmatrix} 0 & 0 & 0 \\ 0 & 25 & 0\\ 0 & 0 & 0 \end{bmatrix}" />

**Conclusion**
- Output of the Center Image is increased with a value which is **same as that of the image**
- It further implies that if we calculate gradient wrt weights of kernel/filter we can get the INput image in the Output as well

### Mathematical form of Convolution (Forward Propagation)
 <img src="https://github.com/sbhrwl/ComputerVision/blob/main/artifacts/images/conv_example.jpg" width=800>

**Output Image(i,j) = Conv(Image, Filter)**

<img src="https://latex.codecogs.com/svg.image?\sum_{x=0}^{3}\sum_{y=0}^{3}&space;Image(i&plus;x,&space;j&plus;y)\star&space;&space;Filter(x,&space;y)" title="\sum_{x=0}^{3}\sum_{y=0}^{3} Image(i+x, j+y)\star Filter(x, y)" />

**3 channels**

### [Backward Propagation](https://www.youtube.com/watch?v=BvrWiL2fd0M&t=770s)
#### Goal
- Calculate Derivative of Loss of network wrt **weights** of filter/kernel
- Calculate Derivative of Loss of network wrt **Output** Image

#### Steps
- Calculation of Derivative of Loss wrt weights of **filter/kernel** can be represented as below

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;L}{\partial&space;w}&space;=&space;\frac{\partial&space;L}{\partial&space;Filter(x,&space;y)}&space;=&space;\sum_{i}^{}\sum_{j}^{}\frac{\partial&space;L}{\partial&space;OutputImage}&space;\bullet&space;\frac{\partial&space;OutputImage}{\partial&space;Filter}" title="\frac{\partial L}{\partial w} = \frac{\partial L}{\partial Filter(x, y)} = \sum_{i}^{}\sum_{j}^{}\frac{\partial L}{\partial OutputImage} \bullet \frac{\partial OutputImage}{\partial Filter}" />

Here, partial Derivative of Output image wrt filter at (x, y), will give **image at the location x, y** represented as **Image(i+x, j+y)**

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;Output&space;Image}{\partial&space;Filter(x,&space;y)}&space;=&space;Image(i&plus;x,&space;j&plus;y)" title="\frac{\partial Output Image}{\partial Filter(x, y)} = Image(i+x, j+y)" />

and after Back propagation through Max Pooling layer, we had a **Matrix** which is equivalent to Derivative of Loss of network wrt **Output Image**

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;L}{\partial&space;OutputImage}&space;=&space;\frac{\partial&space;L}{\partial&space;x^{`}}" title="\frac{\partial L}{\partial OutputImage} = \frac{\partial L}{\partial x^{`}}" />

- FP is a Convolution
- BP is also a Convolution
