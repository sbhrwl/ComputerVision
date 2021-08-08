from datetime import datetime
import time
# from keras.optimizers import SGD, Adam
from keras_preprocessing.image import ImageDataGenerator
from src.core.utils import get_parameters
from src.core.data_preparation import get_data
from src.core.build_model import get_model
from src.core.setup_callbacks import get_callbacks
from src.core.utils import get_random_eraser
from src.core.plot_learning_curve import plot_training_history


def start_training_custom_dataset(model, train_set, test_set):
    config = get_parameters()
    model_compile_config = config["model_compile"]
    # Step 1: Compile model
    model.compile(loss=model_compile_config["loss"],
                  optimizer=model_compile_config["optimizer"],
                  metrics=model_compile_config["metrics"])

    # Step 2: Setup Callbacks
    callbacks = get_callbacks()

    # Step 3: Setup Training parameters
    model_training_config = config["model_training"]
    batch_size = model_training_config["batch_size"]
    epochs = model_training_config["epochs"]
    steps_per_epoch = model_training_config["steps_per_epoch"]
    validation_steps = model_training_config["validation_steps"]

    # Step 4: Start Training
    history = model.fit(train_set,
                        validation_data=test_set,
                        epochs=epochs,
                        batch_size=batch_size,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        verbose=1,
                        callbacks=callbacks)


def start_training(model, X_train, y_train, X_validation, y_validation):
    config = get_parameters()
    data_augmentation = config["data_augmentation"]
    inception_scratch_training = config["inception_scratch_training"]

    # Setup Train and Validation data
    if data_augmentation == "none":
        print("Performing no data augmentation")
    elif data_augmentation == "preprocess":
        print("Performing preprocess data augmentation")
        train_generator = ImageDataGenerator(
            rotation_range=2,
            horizontal_flip=True,
            zoom_range=.1)

        validation_generator = ImageDataGenerator(
            rotation_range=2,
            horizontal_flip=True,
            zoom_range=.1)
        train_generator.fit(X_train)
        validation_generator.fit(X_validation)
    elif data_augmentation == "random_erasing":
        print("Performing random erasing")
        train_generator = ImageDataGenerator(preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=True))
        validation_generator = ImageDataGenerator(
            preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=True))
        train_generator.fit(X_train)
        validation_generator.fit(X_validation)

    # Step 1: Compile model
    # learn_rate = .001
    # sgd = SGD(lr=learn_rate, momentum=.9, nesterov=False)
    # adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, rmsgrad=False)
    model_compile_config = config["model_compile"]
    model.compile(loss=model_compile_config["loss"],
                  optimizer=model_compile_config["optimizer"],
                  metrics=model_compile_config["metrics"])

    # Step 2: Setup Callbacks
    callbacks = get_callbacks()

    # Step 3: Setup Training parameters
    model_training_config = config["model_training"]
    batch_size = model_training_config["batch_size"]
    epochs = model_training_config["epochs"]
    validation_steps = model_training_config["validation_steps"]

    # Step 4: Start Training
    start = datetime.now()  # time.time()
    if data_augmentation == "none":
        if inception_scratch_training == "Y":
            print("Starting Inception training from scratch")
            history = model.fit(X_train, [y_train, y_train, y_train],
                                validation_data=(X_validation, [y_validation, y_validation, y_validation]),
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=callbacks)
        else:
            history = model.fit(X_train, y_train,
                                validation_data=(X_validation, y_validation),
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=callbacks)
    else:
        history = model.fit(train_generator.flow(X_train, y_train, batch_size=batch_size),
                            validation_data=validation_generator.flow(X_validation, y_validation,
                                                                      batch_size=batch_size),
                            # validation_data=(X_validation, y_validation),
                            epochs=epochs,
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            validation_steps=validation_steps,
                            callbacks=callbacks,
                            verbose=1)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    # print('Training time: %s' % (t - time.time()))

    # Step 5: Save Model
    # model.save("artifacts/model/model.h5")
    print("Model saved to disk via ModelCheckpoint callback")

    # Step 6: Plot Training history
    # plot_training_history(model)


def model_preparation_custom_dataset():
    train_dataset, test_dataset = get_data()
    model = get_model()
    start_training_custom_dataset(model, train_dataset, test_dataset)


def model_preparation():
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = get_data()
    model = get_model()
    start_training(model, train_features, train_labels, validation_features, validation_labels)


if __name__ == '__main__':
    config = get_parameters()
    dataset_config = config["dataset"]
    if dataset_config == "custom_dataset":
        model_preparation_custom_dataset()
    else:
        model_preparation()
