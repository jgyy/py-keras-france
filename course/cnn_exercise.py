"""
Convolutional Neural Networks Exercises
"""
from tensorflow.keras import datasets, utils, models, layers, backend
from matplotlib.pyplot import show, imshow, figure


def main():
    """
    main function
    """
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data("/tmp/mnist.npz")
    print(x_train.shape)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train_cat = utils.to_categorical(y_train, 10)
    y_test_cat = utils.to_categorical(y_test, 10)
    backend.clear_session()
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )
    model.summary()
    model.fit(x_train, y_train_cat, batch_size=128, epochs=2, validation_split=0.3)
    model.evaluate(x_test, y_test_cat)
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    print(x_train.shape)
    figure()
    imshow(x_train[1])
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    print(y_train.shape)
    y_train_cat = utils.to_categorical(y_train, 10)
    y_test_cat = utils.to_categorical(y_test, 10)
    print(y_train_cat.shape)
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            32, (3, 3), padding="same", input_shape=(32, 32, 3), activation="relu"
        )
    )
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )
    model.summary()
    model.fit(
        x_train,
        y_train_cat,
        batch_size=32,
        epochs=2,
        validation_data=(x_test, y_test_cat),
        shuffle=True,
    )


if __name__ == "__main__":
    main()
    show()
