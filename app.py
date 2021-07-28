from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from src.core.utils import decode_image
from src.basic_image_classifier.predict import ImageClassification

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


# @cross_origin()
class ClientApp:
    def __init__(self):
        self.model_input_file = "input_image_for_model.jpg"
        self.classifier = ImageClassification(self.model_input_file)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_route():
    image = request.json['image']
    decode_image(image, clApp.model_input_file)
    result = clApp.classifier.dog_or_cat()
    return jsonify(result)


# port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = ClientApp()
    # app.run(host='0.0.0.0', port=port)
    # app.run(host='127.0.0.1', port=port)
    app.run(host='0.0.0.0', port=8000, debug=True)
