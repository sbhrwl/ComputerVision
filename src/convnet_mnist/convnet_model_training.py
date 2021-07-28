from keras.callbacks import LearningRateScheduler
# from keras.optimizers import adam
from src.convnet_mnist.convnet_model_architectures import model_architecture_1


def build_model():
    model = model_architecture_1()
    return model


# Creating the "scheduler" function with two arguments i.e learning rate and epoch
def scheduler(epoch, lr):
    # Learning rate = Learning rate * 1/(1 + decay * epoch)
    # here decay is 0.319 and epoch is 10.
    return round(0.003 * 1 / (1 + 0.319 * epoch), 10)


def compile_model(model):
    # compile the model
    # optimizer = adam(lr=0.003, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


def start_training(model, X_train, y_train, X_test, y_test):
    # Here we are training our model using the data and
    # using batch size of 128,number of epochs are 20 and
    # using verbose=1 for printing out all the results.
    # In the callbacks parameter we are using the Learning Rate Scheduler

    model.fit(X_train, y_train,
              batch_size=128, epochs=1, verbose=1,
              validation_data=(X_test, y_test),
              callbacks=[LearningRateScheduler(scheduler, verbose=1)])


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


def model_preparation():
    train_features, train_labels, test_features, test_labels = data_preparation()
    model = build_model()
    compile_model(model)
    start_training(model, train_features, train_labels, test_features, test_labels)
    # load_model(model)
    model_accuracy = evaluate_model(model, test_features, test_labels)
    return model_accuracy


if __name__ == '__main__':
    accuracy_of_model = model_preparation()
    print(accuracy_of_model)
