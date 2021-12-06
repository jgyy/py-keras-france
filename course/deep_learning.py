"""
Deep Learning Intro
"""
from os import getcwd
from os.path import join
from numpy import linspace, meshgrid, argmax, c_
from pandas import DataFrame, read_csv
from seaborn import pairplot
from matplotlib.pyplot import show, plot, legend, figure, contourf
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras import models, layers, optimizers, utils

path = lambda x: join(getcwd(), "data", x)


def plot_decision_boundary(model, x_data, y_data):
    """
    plot
    """
    amin, bmin = x_data.min(axis=0) - 0.1
    amax, bmax = x_data.max(axis=0) + 0.1
    hticks = linspace(amin, amax, 101)
    vticks = linspace(bmin, bmax, 101)
    aa_data, bb_data = meshgrid(hticks, vticks)
    ab_data = c_[aa_data.ravel(), bb_data.ravel()]
    c_data = model.predict(ab_data)
    cc_data = c_data.reshape(aa_data.shape)
    figure(figsize=(12, 8))
    contourf(aa_data, bb_data, cc_data, cmap="bwr", alpha=0.2)
    plot(x_data[y_data == 0, 0], x_data[y_data == 0, 1], "ob", alpha=0.5)
    plot(x_data[y_data == 1, 0], x_data[y_data == 1, 1], "xr", alpha=0.5)
    legend(["0", "1"])


def main():
    """
    main function
    """
    x_data, y_data = make_moons(n_samples=1000, noise=0.1, random_state=0)
    plot(x_data[y_data == 0, 0], x_data[y_data == 0, 1], "ob", alpha=0.5)
    plot(x_data[y_data == 1, 0], x_data[y_data == 1, 1], "xr", alpha=0.5)
    legend(["0", "1"])
    print(x_data.shape)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42
    )
    model = models.Sequential()
    model.add(layers.Dense(1, input_shape=(2,), activation="sigmoid"))
    model.compile(
        optimizers.Adam(learning_rate=0.05), "binary_crossentropy", metrics=["accuracy"]
    )
    model.fit(x_train, y_train, epochs=200)
    results = model.evaluate(x_test, y_test)
    print(results)
    print("The Accuracy score on the Train set is:\t{:0.3f}".format(results[1]))
    plot_decision_boundary(model, x_data, y_data)

    model = models.Sequential()
    model.add(layers.Dense(4, input_shape=(2,), activation="tanh"))
    model.add(layers.Dense(2, activation="tanh"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizers.Adam(learning_rate=0.05), "binary_crossentropy", metrics=["accuracy"]
    )
    model.fit(x_train, y_train, epochs=100, verbose=2)
    model.evaluate(x_test, y_test)
    y_train_pred = argmax(x_train, axis=1)
    y_test_pred = argmax(x_test, axis=1)
    print(
        f"The Accuracy score on the Train set is:\t{accuracy_score(y_train, y_train_pred):0.3f}"
    )
    print(
        f"The Accuracy score on the Test set is:\t{accuracy_score(y_test, y_test_pred):0.3f}"
    )
    plot_decision_boundary(model, x_data, y_data)

    dataf = DataFrame(read_csv(path("iris.csv")))
    pairplot(dataf, hue="species")
    x_data = dataf.drop("species", axis=1)
    print(x_data.head())
    target_names = dataf["species"].unique()
    print(target_names)
    target_dict = {n: i for i, n in enumerate(target_names)}
    print(target_dict)
    y_data = dataf["species"].map(target_dict)
    print(y_data.head())
    y_cat = utils.to_categorical(y_data)
    print(y_cat[:10])
    x_train, x_test, y_train, y_test = train_test_split(
        x_data.values, y_cat, test_size=0.2
    )
    model = models.Sequential()
    model.add(layers.Dense(3, input_shape=(4,), activation="softmax"))
    model.compile(
        optimizers.Adam(learning_rate=0.1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, epochs=20, validation_split=0.1)
    y_pred = model.predict(x_test)
    print(y_pred[:5])
    print(classification_report(argmax(y_test, axis=1), argmax(y_pred, axis=1)))
    print(confusion_matrix(argmax(y_test, axis=1), argmax(y_pred, axis=1)))


if __name__ == "__main__":
    main()
    show()
