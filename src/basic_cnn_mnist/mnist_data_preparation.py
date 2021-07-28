from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt


def load_dataset():
    # use Keras to import pre-shuffled MNIST database
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print("The MNIST database has a training set of %d examples." % len(X_train))
    print("The MNIST database has a test set of %d examples." % len(X_test))
    return (X_train, y_train), (X_test, y_test)


def visualise_dataset(features, labels):
    # plot first six training images
    fig = plt.figure(figsize=(20, 20))
    for i in range(6):
        ax = fig.add_subplot(1, 6, i + 1, xticks=[], yticks=[])
        ax.imshow(features[i], cmap='gray')
        ax.set_title(str(labels[i]))


def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y], 2)), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'black')


def preprocess_features(train_features, test_features):
    # rescale to have values within 0 - 1 range [0,255] --> [0,1]
    X_train = train_features.astype('float32') / 255
    X_test = test_features.astype('float32') / 255

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    return X_train, X_test


def preprocess_labels(train_labels, test_labels):
    num_classes = 10
    # print first ten (integer-valued) training labels
    print('Integer-valued labels:')
    print(train_labels[:10])

    # one-hot encode the labels
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(train_labels, num_classes)
    y_test = np_utils.to_categorical(test_labels, num_classes)

    # print first ten (one-hot) training labels
    print('One-hot labels:')
    print(y_train[:10])
    return y_train, y_test


def reshape_features(train_features, test_features):
    # Monochrome image with 1 Channel
    # width * height * channel: 28 * 28 * 1
    # input image dimensions 28x28 pixel images.

    img_rows, img_cols = 28, 28

    X_train = train_features.reshape(train_features.shape[0], img_rows, img_cols, 1)
    X_test = test_features.reshape(test_features.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    print('input_shape: ', input_shape)
    print('x_train shape:', X_train.shape)
    return X_train, X_test
