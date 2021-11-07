# %%
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import PIL
import numpy as np

# %%
## path for train, test, validation set

train_dir = r'C:\Users\jgaur\Tensorflow_Tut\Pneumonia\chest_xray\train'
val_dir = r'C:\Users\jgaur\Tensorflow_Tut\Pneumonia\chest_xray\val'
test_dir = r'C:\Users\jgaur\Tensorflow_Tut\Pneumonia\chest_xray\test'

# %%
# train image
train_img = PIL.Image.open(r"C:\Users\jgaur\Tensorflow_Tut\Pneumonia\chest_xray\train\NORMAL\IM-0162-0001.jpeg")
width, height = train_img.size
print(width, height)
train_img

# %%
classes = ['NORMAL', 'PNEUMONIA']

# %%
## Preprocessing

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, dtype=tf.float32)
val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, dtype=tf.float32)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, dtype=tf.float32)

# path to read the images from directory
train_ds = train_gen.flow_from_directory(train_dir, 
                                    target_size=(128, 128), 
                                    class_mode= 'sparse',  # it will give us the labels as a single integer value
                                    batch_size=64, 
                                    shuffle=True,
                                    color_mode='grayscale', 
                                    classes=classes)

val_ds = val_gen.flow_from_directory(val_dir, 
                                    target_size=(128, 128), 
                                    class_mode='sparse',
                                    batch_size=64, 
                                    shuffle=False, 
                                    color_mode='grayscale',
                                    classes=classes)

test_ds = test_gen.flow_from_directory(test_dir, 
                                    target_size=(128, 128), 
                                    class_mode='sparse',
                                    batch_size=64, 
                                    shuffle=False, 
                                    color_mode='grayscale', 
                                    classes=classes)

# %%
train_batch = train_ds[50]
print(train_batch[1].shape)
print(train_batch[0].shape)

# %%
## plotting some images

def show(batch, pred_labels=None):
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(batch[0][i], cmap='gray')
        lbl = classes[int(batch[1][i])]
        
        if pred_labels is not None:
            lbl += 'Pred/' + classes[int(batch[1][i])]

        plt.xlabel(lbl)

    plt.show()

# %%
show(train_batch)

# %%
## model creation

model = tf.keras.models.Sequential()
# model.add(keras.layers.InputLayer(input_shape=(256, 256, 1)))
model.add(keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(128, 128, 1)))# 32 x 64 x 64
model.add(keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')) # 64, 32, 32
model.add(keras.layers.MaxPool2D((2)))  # 64, 16, 16
model.add(keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'))  # 64 x 8 x 8 
model.add(keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'))  # 64 x 8 x 8 
model.add(keras.layers.MaxPool2D((2)))  # 64 x 4 x 4
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# %%
model.summary()

# %%
loss = keras.losses.BinaryCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy']
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# %%
## training 

epochs = 10

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)

# %%
def plot_loss(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(train_loss, '-rx', label='Train loss')
    plt.plot(val_loss, '-bx', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epochs vs Loss')
    plt.legend()
    plt.show()

# %%
plot_loss(history)

# %%
def accuracy(history):
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(train_acc, label='Train accuracy')
    plt.plot(val_acc, label='Val accuracy')
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.title("Epochs vs Accuracy")
    plt.legend()
    plt.show()

# %%
accuracy(history)

# %%
model.evaluate(test_ds, verbose=2)

# %%
model.predict(test_ds)

# %%


# %%
