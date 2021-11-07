import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import PIL
import numpy

# Dataset: https://www.kaggle.com/somnath796/indian-dance-form-recognition
classes = os.listdir(r"C:\Users\jgaur\Tensorflow_Tut\Image_Classification\indian-classical-dance\dataset\train")

num_classes = len(classes)

PIL.Image.open(
    r"C:\Users\jgaur\Tensorflow_Tut\Image_Classification\indian-classical-dance\dataset\train\kathak"
    r"\kathak_original_1.jpg_4c663055-85c6-4785-a8b2-ca9b972be7af.jpg")

PIL.Image.open(
    r"C:\Users\jgaur\Tensorflow_Tut\Image_Classification\indian-classical-dance\dataset\train\kathak"
    r"\kathak_original_58.jpg_2b6c496e-5800-49b8-a51b-1333ce7703b2.jpg")

image = PIL.Image.open(r"C:\Users\jgaur\Tensorflow_Tut\Image_Classification\indian-classical-dance\dataset\test\80.jpg")
print(image.size)  # The size of an image is 275 x 183

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                             horizontal_flip=True, rotation_range=20,
                                                             shear_range=0.2, zoom_range=0.2)
val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_batch = train_datagen.flow_from_directory(
    r"C:\Users\jgaur\Tensorflow_Tut\Image_Classification\indian-classical-dance\dataset\train",
    target_size=(128, 128),
    class_mode='sparse',  # it will get us label as a single integer value\
    batch_size=64,
    shuffle=True,
    color_mode='rgb',
    classes=classes
)

val_batch = val_datagen.flow_from_directory(
    r"C:\Users\jgaur\Tensorflow_Tut\Image_Classification\indian-classical-dance\dataset\validation",
    target_size=(128, 128),
    class_mode='sparse',  # it will get us label as a single integer value\
    batch_size=64,
    shuffle=False,
    color_mode='rgb',
    classes=classes
)

test_batch = test_datagen.flow_from_directory(
    r"C:\Users\jgaur\Tensorflow_Tut\Image_Classification\indian-classical-dance\dataset\test",
    target_size=(128, 128),
    class_mode='sparse',  # it will get us label as a single integer value\
    batch_size=64,
    shuffle=False,
    color_mode='rgb',
)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\jgaur\Tensorflow_Tut\Image_Classification\indian-classical-dance\dataset\train",
    class_names=classes, label_mode='int',
    color_mode='rgb', batch_size=64, image_size=(128,
                                                 128), shuffle=True, seed=None, validation_split=None)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\jgaur\Tensorflow_Tut\Image_Classification\indian-classical-dance\dataset\validation",
    class_names=classes, label_mode='int',
    color_mode='rgb', batch_size=64, image_size=(128,
                                                 128), shuffle=False, seed=None, validation_split=None)

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     r"C:\Users\jgaur\Tensorflow_Tut\Image_Classification\indian-classical-dance\dataset\test")


for image, label in train_ds.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(classes[label[i]])
        plt.axis('off')

for image, label in train_ds:
    print(image.shape)
    print(label.shape)
    break

"""
# Normalizing the pixel of the image
"""

normalize_layer = keras.layers.experimental.preprocessing.Rescaling(1. / 255)

normalize_ds = train_ds.map(lambda x, y: (normalize_layer(x), y))

# image, label = next(iter(normalize_ds))
# first_image = image[0]
# print(numpy.min(first_image), numpy.max(first_image))

for image, label in normalize_ds.take(1):
    print(numpy.min(image), numpy.max(image))
    break

data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(128, 128, 3)),
    keras.layers.experimental.preprocessing.RandomZoom(0.1),
    keras.layers.experimental.preprocessing.RandomRotation(0.1)
])

plt.figure(figsize=(10, 10))
for image, label in train_ds.take(1):
    for i in range(9):
        augmented_image = data_augmentation(image)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image[i].numpy().astype('uint8'))

model = keras.models.Sequential([
    data_augmentation,
    keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(128, 128, 3)),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(256),
    keras.layers.Dense(num_classes)
])
model.summary()

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy']

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=metrics)

epochs = 15
callbacks = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=2
)

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[callbacks], verbose=2)

val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']

plt.figure(figsize=(10, 10))
plt.plot(val_loss, 'r-o', label='validation_loss')
plt.plot(train_loss, 'b-o', label='train_loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend('Loss vs No. of epochs')
plt.plot()

train_acc = history.history['accuracy']

plt.figure(figsize=(10, 10))
plt.plot(val_acc, 'r-o', label='val_accuracy')
plt.plot(train_acc, 'b-o', label='train_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.show()
