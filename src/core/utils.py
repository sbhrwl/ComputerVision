import yaml
import argparse
import base64
import numpy as np


def get_parameters():
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="parameters.yaml")
    parsed_args = args.parse_args()

    with open(parsed_args.config) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def decode_image(input_image_base64_string, model_input_file_name):
    image_data_decoded = base64.b64decode(input_image_base64_string)
    with open(model_input_file_name, 'wb') as f:
        f.write(image_data_decoded)
        f.close()


def encodeImageIntoBase64(cropped_image_path):
    with open(cropped_image_path, "rb") as f:
        return base64.b64encode(f.read())


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()
        if p_1 > p:
            return input_img
        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            if left + w <= img_w and top + h <= img_h:
                break
        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c
        return input_img
    return eraser
