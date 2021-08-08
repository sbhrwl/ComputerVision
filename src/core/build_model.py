from src.networks.convnet_mnist_architectures import *
from src.networks.vgg_architectures import *
from src.networks.inception_architectures import *
from src.networks.inception_architectures_tf import *
from src.networks.resnet_architectures import *
from src.networks.densenet_architectures import *
from src.networks.efficientnet_architectures import *
from src.core.utils import get_parameters


def get_model():
    config = get_parameters()
    model_config = config["model"]
    classes = config["classes"]

    if model_config == "model_architecture_original":
        model = model_architecture_original()
    elif model_config == "model_architecture_16_channels_replaced_with_8_channels":
        model = model_architecture_16_channels_replaced_with_8_channels()
    elif model_config == "model_architecture_max_pool_after_image_reduced_to_8":
        model = model_architecture_max_pool_after_image_reduced_to_8()
    elif model_config == "model_architecture_one_more_11_conv_to_reduce_dimension_from_16_to_10_with_conv_size_8":
        model = model_architecture_one_more_11_conv_to_reduce_dimension_from_16_to_10_with_conv_size_8()
    elif model_config == "build_model_vgg_16":
        model = build_model_vgg_16()
    elif model_config == "build_model_vgg_19":
        model = build_model_vgg_19()
    elif model_config == "build_vgg_model_transfer_leaning_custom":
        model = build_vgg_model_transfer_leaning_custom()
    elif model_config == "build_vgg_model_vgg16_transfer_learning_cifar":
        model = build_vgg_model_vgg16_transfer_learning_cifar()
    elif model_config == "build_vgg_model_vgg19_transfer_learning_cifar":
        model = build_vgg_model_vgg19_transfer_learning_cifar()
    elif model_config == "build_model_inception":
        model = build_model_inception()
    elif model_config == "inception_transfer_learning":
        model = inception_transfer_learning()
    elif model_config == "inception_transfer_learning_starting_from_mixed_7_layer":
        model = inception_transfer_learning_starting_from_mixed_7_layer()
    elif model_config == "resnet_transfer_learning":
        model = resnet_transfer_learning()
    elif model_config == "resnet_convnet_transfer_learning":
        model = resnet_convnet_transfer_learning()
    elif model_config == "resnet_transfer_learning_skip_bn":
        model = resnet_transfer_learning_skip_bn()
    elif model_config == "resNet50_scratch":
        model = resNet50_scratch()
    elif model_config == "dense_net_transfer_learning":
        model = dense_net_transfer_learning(classes)
    elif model_config == "dense_net_convnet_transfer_learning":
        model = dense_net_convnet_transfer_learning(classes)
    elif model_config == "efficient_net_transfer_learning":
        model = efficient_net_transfer_learning(classes)
    elif model_config == "efficient_net_convnet_transfer_learning":
        model = efficient_net_convnet_transfer_learning(classes)
    return model
