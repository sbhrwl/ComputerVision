import numpy as np
from keras.models import load_model
from keras.preprocessing import image


class ImageClassification:
    def __init__(self, filename):
        self.filename = filename

    def dog_or_cat(self):
        # load model
        model = load_model('artifacts/model/model.h5')

        # summarize model
        # model.summary()
        image_name = self.filename
        test_image = image.load_img(image_name, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'dog'
            return [{"image": prediction}]
        else:
            prediction = 'cat'
            return [{"image": prediction}]
