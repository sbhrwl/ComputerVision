from src.networks.convnet_mnist_architectures import *
from src.networks.vgg_architectures import *
from src.networks.inception_architectures import *
from src.networks.inception_architectures_tf import *
from src.networks.resnet_architectures import *
from src.networks.densenet_architectures import *
from src.networks.efficientnet_architectures import *


def get_model():
    # print("ConvNet MNIST")
    # model = model_architecture_original()
    # model = model_architecture_16_channels_replaced_with_8_channels()
    # model = model_architecture_max_pool_after_image_reduced_to_8()
    # model = model_architecture_one_more_11_conv_to_reduce_dimension_from_16_to_10_with_conv_size_8()

    # print("VGG")
    # model = build_model_vgg_16()
    # model = build_model_vgg_19()
    # model = build_vgg_model_transfer_leaning_custom()
    # model = build_vgg_model_vgg16_transfer_learning_cifar()
    # model = build_vgg_model_vgg19_transfer_learning_cifar()

    # print("Inception")
    # model = build_model_inception()
    # model = inception_transfer_learning()
    # model = inception_transfer_learning_starting_from_mixed_7_layer()

    # print("ResNet")
    # model = resnet_transfer_learning()
    # model = resnet_convnet_transfer_learning()
    # model = resnet_transfer_learning_skip_bn()
    # model = resNet50_scratch()

    print("DenseNet")
    # model = dense_net_transfer_learning()
    model = dense_net_convnet_transfer_learning()

    # print("EfficientNet")
    # model = efficient_net_transfer_learning()
    # model = efficient_net_convnet_transfer_learning()
    return model
