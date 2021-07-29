# General Approach
## Step 1: Data Preparation
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
  
  ## Step 2: Build Model Architectures
  
  ## Step 3: Model Training
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
  
  ## [Task 1: Basic CNN architectures for MNIST datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/basic_cnn_mnist/model_training.py)
  ## [Task 2: Basic CNN architectures for CIFAR10 datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/basic_cnn_cifar10/model_training.py)
  ## [Task 3: Basic Image Classifier](https://github.com/sbhrwl/ComputerVision/blob/main/src/basic_image_classifier/model_training.py)
  ## [Task 4: Convnet architectures for MNIST datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/convnet_mnist/convnet_model_training.py)
  ## [Task 5: VGG architectures for MNIST datatset](https://github.com/sbhrwl/ComputerVision/blob/main/src/vgg/vgg_model_training.py)
