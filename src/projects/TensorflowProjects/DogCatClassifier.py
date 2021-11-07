import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model 
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np 
from glob import glob
import matplotlib.pyplot as plt
# Re-size all the images to this
IMAGE_SIZE = [224, 224]
train_path = r'C:\Users\jgaur\Tensorflow_Tut\TransferLearning\dataset\training_set'
test_path = r'C:\Users\jgaur\Tensorflow_Tut\TransferLearning\dataset\test_set'
# VGG16 model for image classification with weights trained on ImageNet
# we have to use image size of 224, 224, bcz VGG16 was created in such a way that the input image size is actually [224, 244]

# include_top=False, it means that we are telling weather the last layers needs to be added or not
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
# don't train existing weigths
# if we don't do layer.trainable=Flase, then our whole model starts training again and again

for layer in vgg.layers:
    layer.trainable = False
# useful for getting the number of classes

folders = glob(r'C:\Users\jgaur\Tensorflow_Tut\TransferLearning\dataset\training_set\*')
len(folders)
# Flattening the output from the last layer

x = Flatten()(vgg.output)

# last layer or output layer
 
prediction = Dense(1, activation='sigmoid')(x)
"""
# Creating the Mdoel
"""
model = Model(inputs=vgg.input, outputs=prediction)

model.summary()

# VGG 16 : 2 Conv2D layers and 1 MaxPool2D layers occurs 2 times
# and      3 Conv2D layers and 3 MaxPool2D layers occurs 3 timnes 
loss = keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy']

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
class_names = ['cats', 'dogs']
names = ['cat', 'dog']
"""
# Data Augmentation Part
"""
train_datagen = ImageDataGenerator(rescale = 1./255,
                                shear_range = 0.2,
                                zoom_range = 0.2, 
                                horizontal_flip = True)
                                
test_datagen = ImageDataGenerator(rescale = 1./255)

# the flow_from_directory() : generates the batch of randomly transformed images from the directory.

training_set = train_datagen.flow_from_directory(
        r'C:\Users\jgaur\Tensorflow_Tut\TransferLearning\dataset\training_set', 
        target_size=(224, 224),
        class_mode='binary',
        batch_size=32,
        classes=class_names )

test_set = test_datagen.flow_from_directory(
        r'C:\Users\jgaur\Tensorflow_Tut\TransferLearning\dataset\test_set', 
        target_size=(224, 224),
        class_mode='binary',
        batch_size=32, 
        classes=class_names)
batch1 = training_set[1]
print(batch1[1].shape)
def show(batch, pred_labels=None):
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(batch[0][i], cmap=plt.cm.binary)
        lbl = names[int(batch[1][i])]
        
        if pred_labels is not None:
            lbl += 'Pred/' + class_names[int(batch[1][i])]

        plt.xlabel(lbl)

    plt.show()
show(batch1)
len(test_set)
history = model.fit(
    training_set,
    validation_data=test_set, 
    epochs=5,
    verbose=2)
