import matplotlib.pyplot as plt
from src.basic_cnn_cifar10.data_preparation import load_dataset, visualise_dataset, \
    preprocess_features, preprocess_labels
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.basic_cnn_cifar10.model_architectures import model_architecture_1conv_1max_pool


def build_model():
    # model = model_architecture_cifar10_1conv_1max_pool()
    model = model_architecture_1conv_1max_pool()
    # model = model_architecture_2conv_1max_pool()
    # model = model_architecture_3conv_1max_pool()
    return model


def compile_model(model):
    # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


def start_training(model, X_train, y_train, X_test, y_test):
    # train the model
    early_stopping_cb = EarlyStopping(patience=5,
                                      restore_best_weights=True)
    check_pointer = ModelCheckpoint(filepath='model.weights.best.hdf5',
                                    verbose=1,
                                    save_best_only=True)
    # tensorboard_cb = TensorBoard(log_dir="logs")

    (X_train, X_validation) = X_train[5000:], X_train[:5000]
    (y_train, y_validation) = y_train[5000:], y_train[:5000]
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    print(X_validation.shape[0], 'validation samples')

    history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=12,
                        validation_data=(X_validation, y_validation),
                        callbacks=[early_stopping_cb, check_pointer],
                        verbose=2,
                        shuffle=True)


def load_model(model):
    # load the weights that yielded the best validation accuracy
    model.load_weights('model.weights.best.hdf5')


def evaluate_model(model, X_test, y_test):
    # evaluate test accuracy
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracy = 100 * score[1]

    # print test accuracy
    print('Test accuracy: %.4f%%' % accuracy)
    return accuracy


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
    X_train, y_train, X_test, y_test = data_preparation()
    model = build_model()
    compile_model(model)
    start_training(model, X_train, y_train, X_test, y_test)
    load_model(model)
    model_accuracy = evaluate_model(model, X_test, y_test)
    return model_accuracy


if __name__ == '__main__':
    accuracy_of_model = model_preparation()
    print(accuracy_of_model)
