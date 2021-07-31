from datetime import datetime
from src.inception.data_preparation import data_preparation, data_preparation_32
from src.inception.tf_inception_transfer_learning_model_architectures import *
from src.core.plot_learning_curve import plot_learning_curve


def build_model():
    # model = inception_transfer_learning()
    model = inception_transfer_learning_starting_from_mixed_7_layer()
    return model


def compile_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


def start_training(model, X_train, y_train, X_test, y_test):
    start = datetime.now()
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        steps_per_epoch=10,
                        # steps_per_epoch=50000 // 64, epochs=2,
                        epochs=1,
                        batch_size=32
                        )
    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    model.save("artifacts/model/inception/transfer_learning_model_cifar10.h5")
    print("Model saved to disk")
    plot_learning_curve(history)


def model_preparation():
    # train_features, train_labels, test_features, test_labels = data_preparation_32()
    train_features, train_labels, test_features, test_labels = data_preparation(128, 128)
    model = build_model()
    compile_model(model)
    start_training(model, train_features, train_labels, test_features, test_labels)


if __name__ == '__main__':
    model_preparation()
