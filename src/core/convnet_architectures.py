from keras import layers, models


def model_architecture_1():
    model = models.Sequential()

    # Channel dimension and Receptive field dimensions Change in opposite direction
    # Channel dimension decreases, Receptive field dimension increases

    # channel dimensions = 26x26x10 and Receptive field = 3x3
    model.add(layers.Conv2D(10, (3, 3),
                            activation='relu',
                            input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    # Next step Receptive field = 5x5 because result of 2 3x3 is 5x5
    # channel dimensions = 24x24x16 and Receptive field = 5x5
    model.add(layers.Conv2D(16, (3, 3),
                            activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    # When using 1x1 size of Receptive field remains same
    # Performing 2D convolution followed Max pooling operation
    # channel dimensions = 24x24x10 and Receptive field = 5x5 as using 1x1 kernel
    model.add(layers.Conv2D(10, (1, 1),
                            activation='relu'))  # 24
    # channel dimensions = 12x12x10 and Receptive field = 10x10
    model.add(layers.MaxPooling2D((2, 2)))  # 12

    # channel dimensions = 10x10x16 and Receptive field = 12x12
    model.add(layers.Conv2D(16, (3, 3),
                            activation='relu'))  # 10
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    # channel dimensions = 8x8x16 and Receptive field = 14x14
    model.add(layers.Conv2D(16, (3, 3),
                            activation='relu'))  # 8
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    # channel dimensions = 8x8x16 and Receptive field = 16x16
    model.add(layers.Conv2D(16, (3, 3),
                            activation='relu'))  # 6
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))

    # using 4x4 kernel to see the complete image
    model.add(layers.Conv2D(10, (4, 4)))  # 4

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    return model
