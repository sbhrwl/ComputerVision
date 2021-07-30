from datetime import datetime
from src.basic_image_classifier.data_preparation import image_preprocessing
from src.basic_image_classifier.model_architectures import build_model


def train_model():
    training_set_images, test_set_images = image_preprocessing()
    model = build_model()
    start = datetime.now()
    model.fit(training_set_images,
              steps_per_epoch=62,
              epochs=1,
              validation_data=test_set_images,
              validation_steps=2000)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    model.save("artifacts/model/basic_image_classifier/model.h5")
    print("Model saved to disk")


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
