import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input


def load_model():
    model = tf.keras.models.load_model("artifacts/model/inception/transfer_learning_model_cifar10.h5")
    return model


def predict_image_class():
    model = load_model()
    print("Classes", model.predict_classes)
    print("Model name", model.name)

    input_img = "http://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane1.png"
    input_img = cv2.imread(input_img)
    input_img = cv2.resize(input_img, dsize=(500, 500), interpolation=cv2.INTER_CUBIC)
    print('Input Dimensions - Image : ', input_img.shape)
    cv2.imshow(input_img)

    input_img = cv2.resize(input_img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    x = image.img_to_array(input_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    class_index = np.argmax(predictions[0])
    print(predictions[0], class_index)


if __name__ == '__main__':
    predict_image_class()
