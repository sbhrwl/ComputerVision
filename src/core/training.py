from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from src.core.model_architectures import model_architecture_1conv_1max_pool, model_architecture_2conv_1max_pool, \
    model_architecture_3conv_1max_pool


def build_model():
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

    history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=12,
                        validation_data=(X_test, y_test),
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
