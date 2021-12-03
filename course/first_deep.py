"""
First Deep Learning Model
"""
from numpy import linspace, meshgrid, c_
from matplotlib.pyplot import figure, show, plot, xlim, ylim, legend, title, contourf
from sklearn.datasets import make_circles
from tensorflow.keras import models, layers, optimizers


def main():
    """
    main function
    """
    x_array, y_array = make_circles(
        n_samples=1000, noise=0.1, factor=0.2, random_state=0
    )
    print(x_array)
    print(x_array.shape)

    figure(figsize=(5, 5))
    plot(x_array[y_array == 0, 0], x_array[y_array == 0, 1], "ob", alpha=0.5)
    plot(x_array[y_array == 1, 0], x_array[y_array == 1, 1], "xr", alpha=0.5)
    xlim(-1.5, 1.5)
    ylim(-1.5, 1.5)
    legend(["0", "1"])
    title("Blue circles and Red crosses")

    model = models.Sequential()
    model.add(layers.Dense(4, input_shape=(2,), activation="tanh"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizers.SGD(learning_rate=0.5), "binary_crossentropy", metrics=["accuracy"]
    )
    model.fit(x_array, y_array, epochs=20)
    hticks = linspace(-1.5, 1.5, 101)
    vticks = linspace(-1.5, 1.5, 101)
    aa_array, bb_array = meshgrid(hticks, vticks)
    ab_array = c_[aa_array.ravel(), bb_array.ravel()]
    c_array = model.predict(ab_array)
    cc_array = c_array.reshape(aa_array.shape)

    figure(figsize=(5, 5))
    contourf(aa_array, bb_array, cc_array, cmap="bwr", alpha=0.2)
    plot(x_array[y_array == 0, 0], x_array[y_array == 0, 1], "ob", alpha=0.5)
    plot(x_array[y_array == 1, 0], x_array[y_array == 1, 1], "xr", alpha=0.5)
    xlim(-1.5, 1.5)
    ylim(-1.5, 1.5)
    legend(["0", "1"])
    title("Blue circles and Red crosses")


if __name__ == "__main__":
    main()
    show()
