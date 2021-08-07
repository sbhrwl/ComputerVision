from src.vgg.model_architectures import model_architectures
# from src.inception.model_architectures import model_architectures
# from src.resnet.model_architectures import model_architectures
# from src.densenet.model_architectures import model_architectures
# from src.efficientnet.model_architectures import model_architectures


def build_model():
    model = model_architectures()
    return model
