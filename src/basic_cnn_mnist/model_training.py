from datetime import datetime
from src.basic_cnn_mnist.data_preparation import data_preparation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.basic_cnn_mnist.model_architectures import model_architecture


def build_model():
    model = model_architecture()
    return model


def compile_model(model):
    # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


def start_training(model, X_train, y_train):
    # train the model
    early_stopping_cb = EarlyStopping(patience=5,
                                      restore_best_weights=True)
    check_pointer = ModelCheckpoint(filepath='artifacts/model/basic_cnn_mnist/model.weights.best.hdf5',
                                    verbose=1,
                                    save_best_only=True)
    # tensorboard_cb = TensorBoard(log_dir="logs")

    (X_train, X_validation) = X_train[5000:], X_train[:5000]
    (y_train, y_validation) = y_train[5000:], y_train[:5000]
    print(X_train.shape[0], 'train samples')
    print(X_validation.shape[0], 'validation samples')

    start = datetime.now()
    history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=1,
                        validation_data=(X_validation, y_validation),
                        callbacks=[early_stopping_cb, check_pointer],
                        verbose=2,
                        shuffle=True)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    print("Model saved to disk via ModelCheckpoint callback")


def model_preparation():
    X_train, y_train, X_test, y_test = data_preparation()
    model = build_model()
    compile_model(model)
    start_training(model, X_train, y_train)


if __name__ == '__main__':
    model_preparation()
