from keras import layers, models


# Create a pure CNN with less than 10K training parameters for MNIST dataset having an accuracy>99.40%




# Accuracy 86%, parameters 11.5K
def model_architecture_original():
    model = models.Sequential()

    # Channel dimension and Receptive field dimensions Change in opposite direction
    # Channel dimension decreases, Receptive field dimension increases

    model.add(layers.Conv2D(10, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    # Output channel dimensions = 26x26x10 and Receptive field = 3x3

    model.add(layers.Conv2D(16, (3, 3),  activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    # Next step Receptive field = 5x5 because result of 2 3x3 is 5x5
    # Output channel dimensions = 24x24x16 and Receptive field = 5x5

    model.add(layers.Conv2D(10, (1, 1), activation='relu'))  # 24
    # When using 1x1 size of Receptive field remains same
    # Performing 2D convolution followed Max pooling operation
    # Output channel dimensions = 24x24x10 and Receptive field = 5x5 as using 1x1 kernel

    model.add(layers.MaxPooling2D((2, 2)))  # 12
    # Output channel dimensions = 12x12x10 and Receptive field = 10x10

    model.add(layers.Conv2D(16, (3, 3), activation='relu'))  # 10
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    # Output channel dimensions = 10x10x16 and Receptive field = 12x12

    model.add(layers.Conv2D(16, (3, 3), activation='relu'))  # 8
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    # Output channel dimensions = 8x8x16 and Receptive field = 14x14

    model.add(layers.Conv2D(16, (3, 3), activation='relu'))  # 6
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    # Output channel dimensions = 8x8x16 and Receptive field = 16x16

    # using 4x4 kernel to see the complete image
    model.add(layers.Conv2D(10, (4, 4)))  # 4

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    return model


# Accuracy 93.72%, parameters 6340
def model_architecture_16_channels_replaced_with_8_channels():
    model = models.Sequential()

    model.add(layers.Conv2D(10, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(12, (3, 3),  activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(8, (1, 1), activation='relu'))  # 24
    model.add(layers.MaxPooling2D((2, 2)))  # 12

    model.add(layers.Conv2D(8, (3, 3), activation='relu'))  # 10
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(8, (3, 3), activation='relu'))  # 8
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(10, (4, 4)))  # 4

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    return model


# Accuracy 95.88%, parameters 8K
def model_architecture_max_pool_after_image_reduced_to_8():
    model = models.Sequential()

    model.add(layers.Conv2D(10, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(16, (3, 3),  activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(10, (1, 1), activation='relu'))  # 24
    model.add(layers.MaxPooling2D((2, 2)))  # 12

    model.add(layers.Conv2D(16, (3, 3), activation='relu'))  # 10
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(16, (3, 3), activation='relu'))  # 8
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    # Added a Max pool layer after image dimensions has been reduced to 8
    model.add(layers.MaxPooling2D((2, 2)))  # 4

    model.add(layers.Conv2D(10, (4, 4)))  # 4

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    return model


# Accuracy 95.45%, parameters 6332, epoch 1
# Accuracy 98.14%, parameters 6332, epoch 2
def model_architecture_one_more_11_conv_to_reduce_dimension_from_16_to_10_with_conv_size_8():
    model = models.Sequential()

    model.add(layers.Conv2D(10, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(8, (3, 3),  activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(10, (1, 1), activation='relu'))  # 24
    model.add(layers.MaxPooling2D((2, 2)))  # 12

    model.add(layers.Conv2D(8, (3, 3), activation='relu'))  # 10
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    # Added a 1x1 to reduce dimension from 16 to 10
    model.add(layers.Conv2D(10, (1, 1), activation='relu'))  # 10

    model.add(layers.Conv2D(8, (3, 3), activation='relu'))  # 8
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    # using 4x4 kernel to see the complete image
    model.add(layers.Conv2D(10, (4, 4)))  # 4

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    return model
