# Tasks
- [General Approach](#general-approach)
- [Task 1 Basic CNN architectures for MNIST datatset](#task-1-basic-cnn-architectures-for-mnist-datatset)
- [Task 2 Basic CNN architectures for CIFAR10 datatset](#task-2-basic-cnn-architectures-for-cifar10-datatset)
- [Task 3 Basic CNN architectures for Flower datatset](#task-3-basic-cnn-architectures-for-flower-datatset)
- [Task 4 Basic Image Classifier](#task-4-basic-image-classifier)
- [Task 5 Convnet architectures for MNIST datatset](#task-5-convnet-architectures-for-mnist-datatset)
- [Task 6 VGG architectures for CIFAR10 datatset](#task-6-vgg-architectures-for-cifar10-datatset)
- [Task 7 Inception architectures for CIFAR10 datatset](#task-7-inception-architectures-for-cifar10-datatset)
- [Task 8 Resnet architectures for CIFAR10 datatset](#task-8-resnet-architectures-for-cifar10-datatset)

## General Approach
### Step 1: Data Preparation
  - Load dataset
  - Preprocess features
    - Rescale features to have values within 0 - 1 range, ex: [0,255] --> [0,1]
    ```
    X_train = train_features.astype('float32') / 255
    X_test = test_features.astype('float32') / 255
    ```
  - Preprocess labels
    - One Hot Encode Labels
    ```
    y_train = np_utils.to_categorical(train_labels, num_classes)
    y_test = np_utils.to_categorical(test_labels, num_classes)
    ```
  - Reshape features
  ```
    # Monochrome image with 1 Channel
    # width * height * channel: 28 * 28 * 1
    # input image dimensions 28x28 pixel images.

    img_rows, img_cols = 28, 28

    X_train = train_features.reshape(train_features.shape[0], img_rows, img_cols, 1)
    X_test = test_features.reshape(test_features.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
  ```
  
### Step 2: Build Model Architectures
  
### Step 3: Model Training
  - Compile the model
  - Setup Callbacks
    - Early Stopping call back
    - Check pointer call back to save best found weights
  - Split dataset to create Train and Validation sets
  - Start Training
  ```
  history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=1,
                    validation_data=(X_validation, y_validation),
                    callbacks=[early_stopping_cb, check_pointer],
                    verbose=2,
                    shuffle=True)
  ```
* Below **Training times** are on a Computer with 16GB RAM and 2 cores and 4 Logical processors

## [Task 1 Basic CNN architectures for MNIST datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/basic_cnn_mnist/model_training.py)
  | Model | kernel size | Padding | Parameters | Epochs | Batch size | Accuracy | Training time |
  | ----- | ----------- | --------| -----------| ------ | ---------- | -------- | ------------- |
  | 1conv_1max_pool | 3x3 | same | 402,442 | 1 | 32 | 97.82% | 25 sec |
  | 2conv_1max_pool | 3x3 | same | 822,346 | 1 | 32 | 98.08% | 2 min 6 sec |
  | 3conv_1max_pool | 3x3 | same | 1,699,018 | 1 | 32 | 98.60% | 6 min 56 sec |
  | 1conv_1max_pool_dropout | 3x3 | same | 201,498 | 1 | 32 | 96.06% | 17 sec |
  | alternate_1conv_1max_pool | 3x3 | same | 220,234 | 1 | 32 | 98.10% | 44 sec |

## [Task 2 Basic CNN architectures for CIFAR10 datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/basic_cnn_cifar10/model_training.py)
  | Model | kernel size | Padding | Parameters | Epochs | Batch size | Accuracy | Training time |
  | ----- | ----------- | --------| -----------| ------ | ---------- | -------- | ------------- |
  | 1conv_1max_pool | 3x3 | same | 525,898 | 1 | 32 | 54.08% | 32 sec |
  | 2conv_1max_pool | 3x3 | same | 1,068,682 | 1 | 32 | 60.42% | 2 min 9 sec |
  | 3conv_1max_pool | 3x3 | same | 2,258,958 | 1 | 32 | 65.42% | 7 min 48 sec |
  | 1conv_1max_pool_dropout | 3x3 | same | 263,066 | 1 | 32 | 46.30% | 20 sec |
  | alternate_1conv_1max_pool | 3x3 | same | 282,250 | 1 | 32 | 50.64% | 56 sec |

## [Task 3 Basic CNN architectures for Flower datatset](https://colab.research.google.com/drive/1bxCs_T6PbcKh7v9FccGUEBj_rOjh861C?usp=sharing)
  | Model | kernel size | Padding | Parameters | Epochs | Accuracy | Training time |
  | ----- | ----------- | ------- | ---------- | ------ | -------- | ------------- |
  | Alternate 3 conv and Max pool |  3x3 | same | 2,534,885 | 10 | 58.75% | 20 mins |
  | Data Augmentation with Rotation, Flipping and Zoom  |  3x3 | same | 4,032,101 | 10 | 66.16% | 30 mins |
  ```
  data_augmentation=Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal',input_shape=(180,180,3)),
    layers.experimental.preprocessing.RandomFlip('vertical',input_shape=(180,180,3)),
    layers.experimental.preprocessing.RandomZoom(0.3),
    layers.experimental.preprocessing.RandomRotation(0.4)
  ])
  ```
  
## [Task 4 Basic Image Classifier](https://github.com/sbhrwl/ComputerVision/blob/main/src/basic_image_classifier/model_training.py)
  | Model | kernel size | Padding | Parameters | Epochs | Accuracy | Training time |
  | ----- | ----------- | ------- | ---------- | ------ | -------- | ------------- |
  | Alternate 3 conv and Max pool |  3x3 | same | 646,273 | 1 | 50.00% | 13 sec |

## [Task 5 Convnet architectures for MNIST datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/convnet_mnist/model_training.py)
  | Changes to Existing Model | Parameters | Epochs | Batch size | Accuracy | Training time |
  | ------------------------- | ---------- | -------| ---------- | -------- | ------------- |
  | Original | 11,450 | 1 | 128 | 86.65% | 1 min 39 sec |
  | Replaced 16 channels with 8 in the Conv layers | 6,340 | 1 | 128 | 93.72% | 1 min 26 sec |
  | Added a Max pool layer after image dimensions has been reduced to 8 | 8,298 | 1 | 128 | 95.88% | 1 min 33 sec |
  | Added a 1x1 to reduce dimension from 16 to 10 | 6,332 | 1 | 128 | 95.45% | 1 min 21 sec |
  | Added a 1x1 to reduce dimension from 16 to 10 | 6,332 | 2 | 128 | 98.14% | 2 min 20 sec |
  
  ### Callback: Learing Rate Scheduler
  ```
  from keras.callbacks import LearningRateScheduler
  
  def scheduler(epoch, lr):
    # Learning rate = Learning rate * 1/(1 + decay * epoch)
    # here decay is 0.319 and epoch is 10.
    return round(0.003 * 1 / (1 + 0.319 * epoch), 10)
  
  model.fit(X_train, y_train,
          batch_size=128, epochs=1, verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[LearningRateScheduler(scheduler, verbose=1)])
  ```
  
## [Task 6 VGG architectures for CIFAR10 datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/vgg/model_training.py)
Read comments mentioned under "Usage" in the vgg_model_training.py

  | VGG Model | Parameters | Epochs | Batch size | Accuracy | Training time |
  | --------- | ---------- | -------| -----------| ---------| ------------- |
  | Transfer Learning Custom Dataset | 50,178 out of 14,719,818 | 1 | 1000 | 53.08% | 7 mins 6 sec |
  | Transfer Learning CIFAR10 Dataset | 5,130 out of 14,719,818 | 1 | 1000 | 11.68% | 6 mins 51 sec |
  | VGG 16 scratch CIFAR10 Dataset | 33,638,218 | 1 | 1000 | 10.5% | 58 mins |
  | VGG 19 scratch CIFAR10 Dataset | 38,947,914 | 1 | 1000 |  10% | 1 hour 17 mins |

  ```
  from keras.callbacks import ReduceLROnPlateau
  lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
  callbacks = [lr_reducer]
  ```
  
  - Training on CIFAR10 with image size 224x224 gives below message and then process terminates
  ```
  W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 12845056000 exceeds 10% of free system memory.
  ```
  
## [Task 7 Inception architectures for CIFAR10 datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/inception/model_training.py)  
  | Inception Model | Parameters | Epochs | Batch size | Accuracy | Training time |
  | ----------------| ---------- | -------| -----------| -------- | ------------- |
  | Transfer Learning CIFAR10 Dataset | 31,214,954 | 1 | 32 | 10% | 19 mins |
  | Transfer Learning CIFAR10 Dataset Layers added after mixed 7 layer | 28,322,826 | 1 | 32 | 54.60% | 29 sec |
  | [Inception scratch](https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/) | 7,188,302 | 1 | 256 | 10.43% | 4 min 41 sec |
  
  - [Inception scratch training](https://colab.research.google.com/drive/10OMWzHiPGZA55PMUTJTFt8sn4T_p0hU5?usp=sharing)
    - Change 1:
      ```
      x = AveragePooling2D(pool_size=(7,7), strides=1, padding='valid',name='avg_pool_5_3x3/1')(x)
      ```
      To
      ```
      x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
      ```
    - Change 2: As colab crashes with CIFAR 10 resized to 224x224x3
      ```
      input_layer = Input(shape=(224, 224, 3))
      ```
      To
      ```
      input_layer = Input(shape=(128, 128, 3))
      ```
  
## [Task 8 Resnet architectures for CIFAR10 datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/resnet/model_training.py)
  | Resnetl | Parameters | Epochs | Batch size | Accuracy | Training time |
  | ----------------| ---------- | -------| -----------| -------- | ------------- |
  | Transfer Learning CIFAR10 Dataset | 42,239,102 out of 42,292,222| 1 | 100 | 10.20% | 28 mins 36 sec |
  | Transfer Learning (convnet) CIFAR10 Dataset | 20,490 out of 23,608,202| 1 | 100 | 26.38% | 10 mins 12 sec |
  | [Scratch training CIFAR10 Dataset](https://pylessons.com/Keras-ResNet-tutorial/) | 23,555,082 | 1 | 100 | 10.47% | 10 min 11 sec |
  | Transfer Learning CIFAR100 Dataset | 603,876 out of 24,138,980 | 1 | 100 | 59.13% | 5 hours 42 min 44 sec |
  - Longest Training 
  <img src="https://github.com/sbhrwl/ComputerVision/blob/main/artifacts/images/resnet_cifar100_training.png">
