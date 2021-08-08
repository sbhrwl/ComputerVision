import tensorflow as tf
from tensorflow.keras import datasets, layers, models


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
    return classifier
