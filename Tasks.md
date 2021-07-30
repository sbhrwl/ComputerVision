# Tasks
- [General Approach](#general-approach)
- [Task 7 Inception architectures for CIFAR10 datatset](#task-7-inception-architectures-for-cifar10-datatset)
- [Task 6 VGG architectures for CIFAR10 datatset](#task-6-vgg-architectures-for-cifar10-datatset)
- [Task 5 Convnet architectures for MNIST datatset](#task-5-convnet-architectures-for-mnist-datatset)
- [Task 4 Basic Image Classifier](#task-4-basic-image-classifier)
- [Task 3 Basic CNN architectures for Flower datatset](#task-3-basic-cnn-architectures-for-flower-datatset)
- [Task 2 Basic CNN architectures for CIFAR10 datatset](#task-2-basic-cnn-architectures-for-cifar10-datatset)
- [Task 1 Basic CNN architectures for MNIST datatset](#task-1-basic-cnn-architectures-for-mnist-datatset)

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
## [Task 7 Inception architectures for CIFAR10 datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/inception/inception_model_training.py)  
  | Inception Model | Parameters | Epochs | Batch size | Accuracy | Training time |
  | ----------------| ---------- | -------| -----------| -------- | ------------- |
  | [Transfer Learning](https://github.com/sbhrwl/ComputerVision/blob/cf04f951ec58b51f819d93f2a7090425ade7a85a/src/inception/inception_transfer_learning.py) | 31,214,954 | 1 | 32 | 10% | 19 mins
  | Inception scratch | 10k | 1 |  | % |

## [Task 6 VGG architectures for CIFAR10 datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/vgg/vgg_model_training.py)
  | VGG Model | Parameters | Epochs | Batch size | Accuracy | Training time |
  | --------- | ---------- | -------| -----------| ---------| ------------- |
  | Transfer Learning Custom Dataset | 50,178 | 1 | 1000 | 85% | 6 mins |
  | VGG 16 scratch CIFAR10 Dataset | 33,638,218 | 1 | 1000 | 97.5% | 58 mins |
  | VGG 19 scratch CIFAR10 Dataset | 38,947,914 | 1 | 1000 |  10% | 1 hour 17 mins |

## [Task 5 Convnet architectures for MNIST datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/convnet_mnist/convnet_model_training.py)
  | Changes to Existing Model | Parameters | Epochs | Batch size | Accuracy |
  | ------------------------- | ---------- | -------| ---------- | -------- |
  | max_pool_after_image_reduced_to_8 | 8k | 1 | 128 | 92% |
  | one_more_1x1_conv_to_reduce_dimension_from_16_to_10 | 10k | 1 | 128 | 96% |
  | 16_channels_replaced_with_8_channels | 6k | 2 | 128 | 98% |
  | one_more_1x1_conv_to_reduce_dimension_from_16_to_10 and 16_channels_replaced_with_8_channels | 6k | 2 | 128 | 97% |
  

## [Task 4 Basic Image Classifier](https://github.com/sbhrwl/ComputerVision/blob/main/src/basic_image_classifier/model_training.py)
## [Task 3 Basic CNN architectures for Flower datatset](https://colab.research.google.com/drive/1bxCs_T6PbcKh7v9FccGUEBj_rOjh861C?usp=sharing)
## [Task 2 Basic CNN architectures for CIFAR10 datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/basic_cnn_cifar10/model_training.py)
## [Task 1 Basic CNN architectures for MNIST datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/basic_cnn_mnist/model_training.py)
