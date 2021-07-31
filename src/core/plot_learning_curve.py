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
