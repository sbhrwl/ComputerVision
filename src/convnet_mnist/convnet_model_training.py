from keras.callbacks import LearningRateScheduler
# from keras.optimizers import adam
from src.convnet_mnist.data_preparation import data_preparation
from src.convnet_mnist.convnet_model_architectures import model_architecture


def build_model():
    model = model_architecture()
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
              batch_size=128, epochs=2, verbose=1,
              validation_data=(X_test, y_test),
              callbacks=[LearningRateScheduler(scheduler, verbose=1)])

    model.save("artifacts/model/convnet_mnist/model.h5")
    print("Model saved to disk")


def model_preparation():
    train_features, train_labels, test_features, test_labels = data_preparation()
    model = build_model()
    compile_model(model)
    start_training(model, train_features, train_labels, test_features, test_labels)


if __name__ == '__main__':
    model_preparation()

