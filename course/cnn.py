"""
Convolutional Neural Networks
"""
from numpy import tensordot, array, convolve, ones
from numpy.random import randint
from scipy.signal import convolve2d
from scipy import misc
from tensorflow.keras import datasets, utils, models, layers, backend
from matplotlib.pyplot import show, imshow, plot, legend, title, xlabel, figure, subplot


def main():
    """
    main function
    """
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data("/tmp/mnist.npz")
    print(x_train.shape)
    print(x_test.shape)
    print(x_train[0])
    imshow(x_train[0], cmap="gray")
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)
    print(x_train.shape)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255.0
    x_test /= 255.0
    print(x_train[0])
    y_train_cat = utils.to_categorical(y_train)
    y_test_cat = utils.to_categorical(y_test)
    print(y_train[0])
    print(y_train_cat[0])
    print(y_train_cat.shape)
    print(y_test_cat.shape)
    backend.clear_session()
    model = models.Sequential()
    model.add(layers.Dense(512, input_dim=28 * 28, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )
    hist = model.fit(
        x_train, y_train_cat, batch_size=128, epochs=10, validation_split=0.3
    )
    plot(hist.history["accuracy"])
    plot(hist.history["val_accuracy"])
    legend(["Training", "Validation"])
    title("Accuracy")
    xlabel("Epochs")
    test_accuracy = model.evaluate(x_test, y_test_cat)[1]
    print(test_accuracy)
    a_tensor = randint(10, size=(2, 3, 4, 5))
    b_tensor = randint(10, size=(2, 3))
    print(a_tensor)
    print(a_tensor[0, 1, 0, 3])
    print(b_tensor)
    img = randint(255, size=(4, 4, 3), dtype="uint8")
    print(img)
    figure(figsize=(5, 5))
    subplot(221)
    imshow(img)
    title("All Channels combined")
    subplot(222)
    imshow(img[:, :, 0], cmap="Reds")
    title("Red channel")
    subplot(223)
    imshow(img[:, :, 1], cmap="Greens")
    title("Green channel")
    subplot(224)
    imshow(img[:, :, 2], cmap="Blues")
    title("Blue channel")
    print(2 * a_tensor)
    print(a_tensor + a_tensor)
    print(a_tensor.shape)
    print(b_tensor.shape)
    print(tensordot(a_tensor, b_tensor, axes=([0, 1], [0, 1])))
    print(tensordot(a_tensor, b_tensor, axes=([0], [0])).shape)
    a_array = array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype="float32")
    b_array = array([-1, 1], dtype="float32")
    c_array = convolve(a_array, b_array)
    print(a_array)
    print(b_array)
    print(c_array)
    subplot(211)
    plot(a_array, "o-")
    subplot(212)
    plot(c_array, "o-")
    img = misc.ascent()
    print(img.shape)
    imshow(img, cmap="gray")
    h_kernel = array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    imshow(h_kernel, cmap="gray")
    res = convolve2d(img, h_kernel)
    imshow(res, cmap="gray")
    print(img.shape)
    figure(figsize=(5, 5))
    imshow(img, cmap="gray")
    img_tensor = img.reshape((1, 512, 512, 1))
    model = models.Sequential()
    model.add(layers.Conv2D(1, (3, 3), strides=(2, 1), input_shape=(512, 512, 1)))
    model.compile("adam", "mse")
    img_pred_tensor = model.predict(img_tensor)
    print(img_pred_tensor.shape)
    img_pred = img_pred_tensor[0, :, :, 0]
    imshow(img_pred, cmap="gray")
    weights = model.get_weights()
    print(weights[0].shape)
    imshow(weights[0][:, :, 0, 0], cmap="gray")
    weights[0] = ones(weights[0].shape)
    model.set_weights(weights)
    img_pred_tensor = model.predict(img_tensor)
    img_pred = img_pred_tensor[0, :, :, 0]
    imshow(img_pred, cmap="gray")
    model = models.Sequential()
    model.add(layers.Conv2D(1, (3, 3), input_shape=(512, 512, 1), padding="same"))
    model.compile("adam", "mse")
    img_pred_tensor = model.predict(img_tensor)
    print(img_pred_tensor.shape)
    model = models.Sequential()
    model.add(layers.MaxPool2D((5, 5), input_shape=(512, 512, 1)))
    model.compile("adam", "mse")
    img_pred = model.predict(img_tensor)[0, :, :, 0]
    imshow(img_pred, cmap="gray")
    model = models.Sequential()
    model.add(layers.AvgPool2D((5, 5), input_shape=(512, 512, 1)))
    model.compile("adam", "mse")
    img_pred = model.predict(img_tensor)[0, :, :, 0]
    imshow(img_pred, cmap="gray")
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    print(x_train.shape)
    backend.clear_session()
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Activation("relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )
    model.summary()
    model.fit(x_train, y_train_cat, batch_size=128, epochs=2, validation_split=0.3)
    model.evaluate(x_test, y_test_cat)


if __name__ == "__main__":
    main()
    show()
