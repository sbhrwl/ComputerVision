import matplotlib.pyplot as plt
from src.basic_cnn_cifar10.cifar_10_data_preparation import load_dataset, visualise_dataset, preprocess_features, preprocess_labels
from src.basic_cnn_mnist.training import build_model, compile_model, start_training, load_model, evaluate_model


def data_preparation():
    (train_features, train_labels), (test_features, test_labels) = load_dataset()
    visualise_dataset(train_features)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    engineered_train_features, engineered_test_features = preprocess_features(train_features,
                                                                              test_features)
    engineered_train_labels, engineered_test_labels = preprocess_labels(train_labels, test_labels)
    return engineered_train_features, engineered_train_labels, engineered_test_features, engineered_test_labels


def model_preparation():
    model = build_model()
    compile_model(model)
    start_training(model, X_train, y_train, X_test, y_test)
    load_model(model)
    model_accuracy = evaluate_model(model, X_test, y_test)
    return model_accuracy


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = data_preparation()
    accuracy = model_preparation()
    print(accuracy)
