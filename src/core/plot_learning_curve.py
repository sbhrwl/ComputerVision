import matplotlib.pyplot as plt


def plot_learning_curve(history):
    print("Plotting learning curve")
    print(history.history)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.savefig("artifacts/model/inception/learning_curve.png")
    # plt.show()


def plot_training_history(model):
    # Plot Training history
    f, ax = plt.subplots(2, 1)  # Creates 2 subplots under 1 column

    # Assign the first subplot to graph training loss and validation loss
    ax[0].plot(model.history.history['loss'], color='b', label='Training Loss')
    ax[0].plot(model.history.history['val_loss'], color='r', label='Validation Loss')

    # Next lets plot the training accuracy and validation accuracy
    ax[1].plot(model.history.history['accuracy'], color='b', label='Training  Accuracy')
    ax[1].plot(model.history.history['val_accuracy'], color='r', label='Validation Accuracy')
    plt.savefig("artifacts/model/resnet/learning_curve.png")
    plt.show()
