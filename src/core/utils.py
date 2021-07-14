import base64


def decode_image(input_image_base64_string, model_input_file_name):
    image_data_decoded = base64.b64decode(input_image_base64_string)
    with open(model_input_file_name, 'wb') as f:
        f.write(image_data_decoded)
        f.close()


def encodeImageIntoBase64(cropped_image_path):
    with open(cropped_image_path, "rb") as f:
        return base64.b64encode(f.read())
