from keras.preprocessing.image import ImageDataGenerator


def image_preprocessing():
    train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                              shear_range=0.2,
                                              zoom_range=0.2,
                                              horizontal_flip=True)
    test_data_generator = ImageDataGenerator(rescale=1. / 255)

    training_set = train_data_generator.flow_from_directory('images/train',
                                                            target_size=(64, 64),
                                                            batch_size=32,
                                                            class_mode='binary')  # category

    test_set = test_data_generator.flow_from_directory('images/validation',
                                                       target_size=(64, 64),
                                                       batch_size=32,
                                                       class_mode='binary')
    return training_set, test_set
