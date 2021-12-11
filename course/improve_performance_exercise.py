"""
Improving Performance Exercises
"""
from os import getcwd
from os.path import join
from itertools import islice
from numpy import array, concatenate, argwhere, argmax
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras import (
    models,
    layers,
    backend,
    utils,
    preprocessing,
    datasets,
)
from matplotlib.pyplot import (
    show,
    imshow,
    plot,
    legend,
    figure,
    fill_between,
    ylim,
    title,
    xlabel,
    ylabel,
)


def repeated_training_reg_dropout(x_train, y_train, x_test, y_test, do_dropout=False):
    """
    repeated training regular dropout function
    """
    rate = 0.3
    kernel_regularizer = "l2"
    epochs = 10
    repeats = 3
    histories = []
    for repeat in range(repeats):
        backend.clear_session()
        model = models.Sequential()
        model.add(
            layers.Dense(
                512,
                input_shape=x_train.shape[1:],
                kernel_initializer="normal",
                kernel_regularizer=kernel_regularizer,
                activation="sigmoid",
            )
        )
        if do_dropout:
            model.add(layers.Dropout(rate))
        model.add(
            layers.Dense(
                512,
                kernel_initializer="normal",
                kernel_regularizer=kernel_regularizer,
                activation="sigmoid",
            )
        )
        if do_dropout:
            model.add(layers.Dropout(rate))
        model.add(
            layers.Dense(
                512,
                kernel_initializer="normal",
                kernel_regularizer=kernel_regularizer,
                activation="sigmoid",
            )
        )
        if do_dropout:
            model.add(layers.Dropout(rate))
        model.add(layers.Dense(10, activation="softmax"))
        model.compile("sgd", "categorical_crossentropy", metrics=["accuracy"])
        hist = model.fit(
            x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=0
        )
        histories.append([hist.history["accuracy"], hist.history["val_accuracy"]])
        print(repeat, end=" ")
    histories = array(histories)
    mean_acc = histories.mean(axis=0)
    std_acc = histories.std(axis=0)
    print()
    return mean_acc[0], std_acc[0], mean_acc[1], std_acc[1]


def plot_mean_std(mean, std):
    """
    plot mean standard deviation function
    """
    figure()
    plot(mean)
    fill_between(range(len(mean)), mean - std, mean + std, alpha=0.1)


def main():
    """
    main function
    """
    max_features = 20000
    (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(
        "/tmp/imdb.npz", num_words=max_features, start_char=1, oov_char=2, index_from=3
    )
    print(x_train.shape)
    maxlen = 80
    x_train_pad = preprocessing.sequence.pad_sequences(
        x_train, maxlen=maxlen, truncating="post"
    )
    x_test_pad = preprocessing.sequence.pad_sequences(
        x_test, maxlen=maxlen, truncating="post"
    )
    model = models.Sequential()
    # model.add(layers.Embedding(max_features, 128))
    # model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(x_train[0])
    model.fit(x_train_pad, y_train, batch_size=32, epochs=2, validation_split=0.3)
    score, acc = model.evaluate(x_test_pad, y_test)
    print("Test score:", score)
    print("Test accuracy:", acc)

    exercise2()


def exercise2():
    """
    exercise 2 function
    """
    digits = load_digits()
    x_data, y_data = digits["data"], digits["target"]
    y_cat = utils.to_categorical(y_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_cat, test_size=0.3)
    mean_acc, std_acc, mean_acc_val, std_acc_val = repeated_training_reg_dropout(
        x_train, y_train, x_test, y_test, do_dropout=False
    )
    (
        mean_acc_do,
        std_acc_do,
        mean_acc_val_do,
        std_acc_val_do,
    ) = repeated_training_reg_dropout(x_train, y_train, x_test, y_test, do_dropout=True)
    plot_mean_std(mean_acc, std_acc)
    plot_mean_std(mean_acc_val, std_acc_val)
    plot_mean_std(mean_acc_do, std_acc_do)
    plot_mean_std(mean_acc_val_do, std_acc_val_do)
    ylim(0, 1.01)
    title("Dropout and Regularization Accuracy")
    xlabel("Epochs")
    ylabel("Accuracy")
    legend(
        [
            "Train",
            "Test",
            "Train with Dropout and Regularization",
            "Test with Dropout and Regularization",
        ],
        loc="best",
    )

    exercise3()


def exercise3():
    """
    exercise 3 function
    """
    backend.clear_session()
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    train_gen = preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    test_gen = preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    train = train_gen.flow_from_directory(
        join(getcwd(), "data", "train"),
        target_size=(64, 64),
        batch_size=16,
        class_mode="binary",
    )
    test = test_gen.flow_from_directory(
        join(getcwd(), "data", "test"),
        target_size=(64, 64),
        batch_size=16,
        class_mode="binary",
    )
    model.fit(
        train,
        steps_per_epoch=800,
        epochs=200,
        validation_data=test,
        validation_steps=200,
    )
    x_test = []
    y_test = []
    for tslice in islice(test, 50):
        x_test.append(tslice[0])
        y_test.append(tslice[1])
    x_test = concatenate(x_test)
    y_test = concatenate(y_test)
    y_pred = model.predict(x_test).ravel()
    y_pred = argmax(y_pred)
    print(argwhere(y_test != y_pred).ravel())
    imshow(x_test[14])


if __name__ == "__main__":
    main()
    show()
