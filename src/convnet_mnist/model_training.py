from keras.callbacks import LearningRateScheduler
from datetime import datetime
# from keras.optimizers import adam
from src.convnet_mnist.data_preparation import data_preparation
from src.convnet_mnist.model_architectures import model_architecture


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
    # In the callbacks parameter we are using the Learning Rate Scheduler

    start = datetime.now()
    model.fit(X_train, y_train,
              batch_size=128, epochs=1, verbose=1,
              validation_data=(X_test, y_test),
              callbacks=[LearningRateScheduler(scheduler, verbose=1)])

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    model.save("artifacts/model/convnet_mnist/model.h5")
    print("Model saved to disk")


def model_preparation():
    train_features, train_labels, test_features, test_labels = data_preparation()
    model = build_model()
    compile_model(model)
    start_training(model, train_features, train_labels, test_features, test_labels)


if __name__ == '__main__':
    model_preparation()

