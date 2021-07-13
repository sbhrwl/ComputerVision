import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator


def image_preproacessing():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('images/train',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')  # category

    test_set = test_datagen.flow_from_directory('images/validation',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
    return training_set, test_set


def build_model():
    # Option 1
    # classifier = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    #     tf.keras.layers.MaxPooling2D((2,2)),
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(64,activation='relu'),
    #     tf.keras.layers.Dense(1,activation='sigmoid')
    #     # tf.keras.layers.Dense(10,activation='softmax')
    # ])

    # Option 2
    classifier = models.Sequential()
    classifier.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    classifier.add(layers.MaxPooling2D((2, 2)))
    classifier.add(layers.Conv2D(64, (3, 3), activation='relu'))
    classifier.add(layers.MaxPooling2D((2, 2)))
    classifier.add(layers.Conv2D(64, (3, 3), activation='relu'))
    classifier.add(layers.Flatten())
    classifier.add(layers.Dense(64, activation='relu'))
    classifier.add(layers.Dense(1, activation='sigmoid'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(classifier.summary())
    return(classifier)


def train_model():
    training_set_images, test_set_images = image_preproacessing()
    model = build_model()
    model.fit_generator(training_set_images,
                         steps_per_epoch=8000,
                         epochs=1,
                         validation_data=test_set_images,
                         validation_steps=2000)

    model.save("artifacts/model/model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    train_model()

# Part 3 - Making new predictions
# import numpy as np
# from keras.preprocessing import image
# from keras.models import load_model
#
# test_image = image.load_img('cat.jpg', target_size=(64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# model = load_model('model.h5')
# result = model.predict(test_image)
# # training_set.class_indices
# if result[0][0] == 1:
#     prediction = 'dog'
#     print(prediction)
# else:
#     prediction = 'cat'
#     print(prediction)
