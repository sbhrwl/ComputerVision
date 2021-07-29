from src.inception.data_preparation import data_preparation_transfer_learning
from src.inception.inception_model_architectures import inception_transfer_learning


def build_model():
    model = inception_transfer_learning()
    return model


def compile_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


def start_training(model, X_train, y_train, X_test, y_test):
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        steps_per_epoch=10,
                        # steps_per_epoch=50000 // 64, epochs=2,
                        epochs=50,
                        batch_size=32
                        )

    model.save("artifacts/model/inception/transfer_learning_model_cifar10.h5")
    print("Model saved to disk")


def model_preparation():
    train_features, train_labels, test_features, test_labels = data_preparation_transfer_learning()
    model = build_model()
    compile_model(model)
    start_training(model, train_features, train_labels, test_features, test_labels)


if __name__ == '__main__':
    model_preparation()
