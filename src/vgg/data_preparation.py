from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


def data_preparation():
    # Give dataset path
    train_path = 'images/train'
    test_path = 'images/validation'

    # Data Augmentation
    train_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Data Augmentation
    test_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Make sure you provide the same target size as initialized for the image size
    train_set = train_data_generator.flow_from_directory('images/train',
                                                         target_size=(224, 224),
                                                         batch_size=32,
                                                         class_mode='categorical')

    test_set = test_data_generator.flow_from_directory('images/validation',
                                                       target_size=(224, 224),
                                                       batch_size=32,
                                                       class_mode='categorical')
    return train_set, test_set
