"""
### In this notebook, we will see how we can use autoencoder as a classifier with fashion MNIST dataset.
- Fashion MNIST dataset is a 28x28 grayscale images of 70,000 fashion products from 10 categories, with 7,000 images per category. The training set has 60,000 images, and the test set has 10,000 images.
- It is a replacement for the original MNIST handwritten digits dataset for producing better results. The Fashion MNIST dataset consists of 10 different classes of fashion accessories like shirts, trousers, sandals, etc 
- The image dimensions, training and test splits are similar to the original MNIST dataset. 
- The dataset is freely available on this URL and can be loaded using both tensorflow and keras as a framework without having to download it on your computer.
"""

"""
So, what we want to do is built a convolutional autoencoder and use the encoder part of it combined with fully connected layers to recognize a new sample from the test set correctly.
"""

# Loading the dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import gzip

from sklearn.model_selection import train_test_split

from keras.models import Model, Sequential
from keras.optimizers import RMSprop, Adam, SGD, Adadelta
from keras.layers import Conv2D, Input, Dense, Flatten,Dropout, merge, Reshape, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras import regularizers
from keras import backend as k
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import itertools

# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

"""
The way we read the data is as follows:
- We open and read the gzip file as bytestream
- We then convert the string stored in buffer using np.frombuffer() to numpy array of type float32.
- Data is then reshaped into a 3-dimentional array.
"""

"""
### Loading the data
"""

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buffer = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28,28)
        return data

train_data = extract_data('train-images-idx3-ubyte.gz' ,60000)
test_data = extract_data('t10k-images-idx3-ubyte.gz',10000)

"""
The way we read the labels is the same as we read the data:
- We define a function which opens the label file as a bytestream using bytestream.read() to which you pass the label dimension as 1 and the total number of images.
- We then convert the string stored in the buffer to a numpy array of type int64.
- We do not need to reshape the array since the variable labels will return a column vector of dimension 60,000 x 1.
"""

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buffer = bytestream.read(1 * num_images)
        labels = np.frombuffer(buffer, dtype = np.uint8).astype(np.uint64)
        return labels

train_labels = extract_labels('train-labels-idx1-ubyte.gz', 60000)
test_labels = extract_labels('t10k-labels-idx1-ubyte.gz',10000)

train_labels[:10]

"""
### Data Exploration
"""

# To see the dimension of images in the training set
print("Training set (images) shape: {shape}", format(train_data.shape))

# To see the dimension of images in the test set
print("Test set (image) shape: {shape}", format(test_data.shape))

"""
We don't need training an testing labels for convolutional encoder as the task at hand is to reconstruct the images. However, we do need them for classification task and for data exploration purposes.
"""

# we will now create a dictionary of class names with their corresponding class labels:
label_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J' 
}

"""
Let us now look at few of the images in the dataset
"""

plt.figure(figsize=[5,5])

# display an image from the training data
plt.subplot(121)
curr_img = np.reshape(train_data[10], (28,28))
curr_label = train_labels[10]
plt.imshow(curr_img, cmap = 'gray')
plt.title("(Label: " + str(label_dict[curr_label]) + ")")

# display an image from the testing data
plt.subplot(122)
curr_img = np.reshape(test_data[10], (28,28))
curr_label = test_labels[10]
plt.imshow(curr_img, cmap= 'gray')
plt.title("(Label: " + str(label_dict[curr_label])+ ")")

"""
These images are assigned class labels as 0 or A / 4 or E. This means all the alphabets having class of E will have the class label of 4.

"""

"""
### Data Preprocessing
"""

"""
We need to do the preprocessing of the data before feeding it to the model as our images are gray scale images having pixel values ranging from 0 to 255 with dimension of 28 * 28.
So we will convert each 28 * 28 image into a matrix of size 28 * 28 * 1 which can be fed to the network.
"""

train_data = train_data.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)
print("Train data shape", train_data.shape)
print("Test data shape", test_data.shape)

# checking the datatype of the numpy arrays
train_data.dtype, test_data.dtype

# rescaling the pixel values to 0-1 scale by rescaling with maximum pixel value of the training and testing data
train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)

# verify the maximum value of the training and the testing data
np.max(train_data), np.max(test_data)

"""
### Splitting the data into train and validation sets
"""

# We will now split the data into 80% training and 20% validation. This will help us to reduce the chances of overfitting
# as you would be validating the model on the data it has not seen in the training.
X_train, X_val, y_train, y_val = train_test_split(train_data, train_data, test_size = 0.2, random_state = 12345)

X_train.shape, X_val.shape

# Now the data is ready to be fed to the network.
batch_size = 128
epochs = 200
channel = 1
x, y = 28, 28
input_image = Input(shape = (x, y, channel))
num_classes = 10

"""
We will build encoder and a decoder using convolutional layers.
Batch Normalization layer is used
"""

# Building an encoder
def encoder(input_image):
    # input dimension - 28 * 28 * 1
    conv1 = Conv2D(32, (3,3), activation = 'relu', padding = 'same') (input_image)  # 28 *28 * 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3,3), activation = 'relu', padding = 'same') (conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size = (2,2))(conv1)  # 14 * 14 * 32
    conv2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same') (pool1) # 14 * 14 * 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same') (conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size = (2,2)) (conv2) # 7 * 7 * 64
    conv3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same') (pool2) # 7 * 7 * 128
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same') (conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256,(3,3), activation = 'relu', padding = 'same') (conv3) # 7 * 7 * 256
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3,3), activation = 'relu', padding = 'same') (conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

# Building a decoder
def decoder(conv4):
    conv5 = Conv2D(128, (3,3), activation = 'relu', padding = 'same') (conv4) # 7 * 7 * 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3,3), activation= 'relu', padding ='same') (conv5)# 7 * 7 * 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3,3), activation= 'relu', padding = 'same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2)) (conv6)  # 14 * 14 * 64
    conv7 = Conv2D(32, (3,3), activation = 'relu', padding = 'same') (up1)# 14 * 14 * 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3,3), activation ='relu', padding = 'same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 * 28 * 32
    decoded = Conv2D(1,(3, 3),  activation = 'sigmoid', padding ='same') (up2) # 28 * 28 * 1
    return decoded

# Compiling the model using RMSProp optimizer (helps in reducing overfitting by
# changing learning rate while the model is training) and specifying the loss as mean squared loss
autoencoder = Model(input_image, decoder(encoder(input_image)))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

# Let us see the autoencoder summary
autoencoder.summary()

"""
### Training the model 
"""

# we will now train the autoencoder for 200 epochs using keras fit function
# We are going to save the history to plot accuracy and loss curves to help in analyzing the model performance.
csv_logger = CSVLogger('autoencoder_log.csv', append = True, separator = ';')
history = autoencoder.fit(X_train, y_train, batch_size = batch_size, epochs= epochs,
                                   verbose =1, validation_data = (X_val, y_val), callbacks = [csv_logger])

# saving the model
autoencoder_json = autoencoder.to_json()
with open('autoencoder_json','w') as json_file:
    json_file.write(autoencoder_json)

# Now we can plot the loss curves to visualize the model performance
# list all data in history
print(history.history.keys())


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

"""
We can clearly see that the training and validation loss are in sync. The validation loss is decreasing which is a good sign which also mean that model is not overfitting. So, we can say the model's generalization capabity is good.

Now, we will use the encoder part of the autoencoder to classify the fashion MNIST images.
"""

# We will need the encoder weights of the autoencoder for classification task. We will now save the
# full autoencoder weights and then we will learn to extract the encoder weights.
autoencoder.save_weights('autoencoder_weights.h5')

"""
### Classifying the fashion MNIST images
"""

"""
- Loading the encoder weights from the trained autoencoder's weight
- We will add a few fully connected (dense) layers to the encoder to classify the fashion MNIST images.
- We will convert labels into one-hot encoding vectors. In one-hot encoding, we convert the categorical values to
    a vector of numbers as machine learning algorithms cannot work with understand categorical data directly.
- So, in our case the one hot encoding will be a row vector and for each image it will have a dimension of 1 X 10.
- The vector consists of all zeros except for the class that it represents, and for that, it is 1.

"""

X_train_one_hot = to_categorical(train_labels)
X_test_one_hot = to_categorical(test_labels)

# Display the change for category label using one-hot encoding
print('Original label:', train_labels[0])
print('One-hot label:', X_train_one_hot[0])

# Splitting the data again into train and test for classification purpose using the one hot labels this time
X_trainc, X_valc, y_trainc, y_valc = train_test_split(train_data, X_train_one_hot, test_size = 0.2, random_state = 12345)

# checking the shape
X_trainc.shape, X_valc.shape, y_trainc.shape, y_valc.shape

# Now we will be using building the classification model using the encoder part (define encoder) of the autoencoder
# defining the fully connected layer for the encoder part
def fc(enco):
    flat = Flatten() (enco)
    fc1 = Dense(128, activation = 'relu') (flat)
    fc2 = Dense(num_classes, activation = 'softmax') (fc1)
    return fc2

encode = encoder(input_image)
full_model = Model(input_image, fc(encode))

for l1, l2 in zip(full_model.layers[:19], autoencoder.layers[:19]):
    l1.set_weights(l2.get_weights())

# Since, we are using the encoder part of the autoencoder for classification purpose, we should make sure that
# the weights of the encoder part of the autoencoder are are similar to the weights you loaded to the 
# encoder function of the classification model. If they are not similar, there is no pointin using the autoencoder classification
# strategy.

# First layer weight of autoencoder
autoencoder.get_weights()[0][1]

full_model.get_weights()[0][1]

"""
The weights are similar, so we can move further and compile the model and start the training
"""

# Since the encoder part is already trained, we do not need to train it again. So, we will make those
# 19 layers of the model as trainable false. We will only train the fully connected layer part
for layer in full_model.layers[:19]:
    layer.trainable = False

# Now, we will compile the model
full_model.compile(loss ='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# summary of the model
full_model.summary()

"""
### Training the classification model
"""

# Training the model for 100 epochs using the keras fit function and visualizing the accuracy and loss curves

history_classification = full_model.fit(X_train, y_trainc, batch_size = 64, epochs= 100,
                                   verbose =1, validation_data = (X_valc, y_valc))

# Saving the classification model's weights
full_model.save_weights('full_model_weights.h5')

"""
### Retraining the full model
"""

# Now we will retrain the full model along with the other 19 layers
for layer in full_model.layers[:19]:
    layer.trainable = True

# Now, we will compile the model
full_model.compile(loss ='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history_classification = full_model.fit(X_train, y_trainc, batch_size = 64, epochs= 100,
                                   verbose =1, validation_data = (X_valc, y_valc))

# Saving the complete model now
full_model.save_weights('full_model_complete.h5')

# plotting the accuracy and loss curves
# Now we can plot the loss curves to visualize the model performance
# list all data in history
print(history_classification.history.keys())

# summarize history for loss
plt.plot(history_classification.history['acc'])
plt.plot(history_classification.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# summarize history for loss
plt.plot(history_classification.history['loss'])
plt.plot(history_classification.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

"""
### Model evaluation on the test set
"""

test_eval = full_model.evaluate(test_data, X_test_one_hot, verbose=1)

# Let us see the test accuracy and loss
print('Test Accuracy', test_eval[0])
print('Test Loss', test_eval[1])

"""
### Prediction
"""

predicted_classes = full_model.predict(test_data)
print(predicted_classes[1])

predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

print(predicted_classes[1])

predicted_classes.shape, test_labels.shape

# finding the correct predicted labels
correct = np.where(predicted_classes == test_labels)[0]

print("Correctly identified samples are: ", len(correct))
for i,correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_data[correct].reshape(28,28), cmap = 'gray', interpolation = 'none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_labels[correct]))
    plt.tight_layout()

# finding the incorrect predictions
incorrect = np.where(predicted_classes != test_labels)[0]

print("Incorrectly identified samples are: ", len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_data[incorrect].reshape(28,28), cmap = 'gray', interpolation = 'none')
    plt.title("Predicted {} class {}".format(predicted_classes[incorrect], test_labels[incorrect]))
    plt.tight_layout()

"""
### Metrics
"""

# Accuracy
print("Accuracy: ", accuracy_score(test_labels, predicted_classes))

# Classification report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_labels, predicted_classes, target_names=target_names))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

np.unique(test_labels)

class_names = np.unique(test_labels)

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_labels, predicted_classes)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

"""
We learned how to use an autoencoder to classify images with python and keras. 
"""