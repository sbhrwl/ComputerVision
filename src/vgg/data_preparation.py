from keras.datasets import cifar10
from keras.utils import np_utils
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


def data_preparation_custom():
    # Give dataset path, size of image 224x224
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


def data_preparation_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    print(X_train.shape)
    print(X_test.shape)
    return X_train, y_train, X_test, y_test
