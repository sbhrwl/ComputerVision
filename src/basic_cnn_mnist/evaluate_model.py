from src.basic_cnn_mnist.data_preparation import data_preparation


def load_model(model):
    # load the weights that yielded the best validation accuracy
    model.load_weights('artifacts/model/basic_cnn_mnist/model.weights.best.hdf5')
    return model


def evaluate_model(model, X_test, y_test):
    # evaluate test accuracy
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracy = 100 * score[1]

    # print test accuracy
    print('Test accuracy: %.4f%%' % accuracy)
    return accuracy


if __name__ == '__main__':
    test_features, test_labels = data_preparation()
    print(test_features.shape[0], 'test samples')
    model_to_evaluate = load_model(model)
    model_accuracy = evaluate_model(model_to_evaluate, test_features, test_labels)
    print(model_accuracy)
