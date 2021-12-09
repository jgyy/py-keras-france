"""
Deep Learning Intro
"""
from os import getcwd
from os.path import join
from numpy import array, dot
from pandas import DataFrame, MultiIndex, read_csv, concat
from seaborn import pairplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import scale
from tensorflow.keras import models, layers, optimizers, backend
from matplotlib.pyplot import (
    show,
    title,
    figure,
    subplot,
    xlabel,
    xlim,
    ylim,
    tight_layout,
    scatter,
)

path = lambda x: join(getcwd(), "data", x)


def main():
    """
    main function
    """
    a_list = array([1, 3, 2, 4])
    print(a_list)
    print(type(a_list))
    a_array = array([[3, 1, 2], [2, 3, 4]])
    b_array = array([[0, 1], [2, 3], [4, 5]])
    c_array = array([[0, 1], [2, 3], [4, 5], [0, 1], [2, 3], [4, 5]])
    print("A is a {} matrix".format(a_array.shape))
    print("B is a {} matrix".format(b_array.shape))
    print("C is a {} matrix".format(c_array.shape))
    print(a_array[0])
    print(c_array[2, 0])
    print(b_array[:, 0])
    print(3 * a_array)
    print(a_array + a_array)
    print(a_array * a_array)
    print(a_array / a_array)
    print(a_array - a_array)
    print(a_array.shape)
    print(b_array.shape)
    print(a_array.dot(b_array))
    print(dot(a_array, b_array))
    print(b_array.dot(a_array))
    print(c_array.shape)
    print(a_array.shape)
    print(c_array.dot(a_array))
    dataf = DataFrame(read_csv(path("banknotes.csv")))
    print(dataf.head())
    print(dataf["class"].value_counts())
    pairplot(dataf, hue="class")
    x_data = scale(dataf.drop("class", axis=1).values)
    y_data = dataf["class"].values
    model = RandomForestClassifier()
    print(cross_val_score(model, x_data, y_data))
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42
    )
    backend.clear_session()
    model = models.Sequential()
    model.add(layers.Dense(1, input_shape=(4,), activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=10)
    result = model.evaluate(x_test, y_test)
    historydf = DataFrame(history.history, index=history.epoch)
    historydf.plot(ylim=(0, 1))
    title("Test accuracy: {:3.1f} %".format(result[1] * 100), fontsize=15)

    learning(x_train, x_test, y_train, y_test)


def learning(x_train, x_test, y_train, y_test):
    """
    learning rates function
    """
    dflist = []
    learning_rates = [0.01, 0.05, 0.1, 0.5]
    for learn in learning_rates:
        backend.clear_session()
        model = models.Sequential()
        model.add(layers.Dense(1, input_shape=(4,), activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.SGD(learning_rate=learn),
            metrics=["accuracy"],
        )
        history = model.fit(x_train, y_train, batch_size=16, epochs=10)
        dflist.append(DataFrame(history.history, index=history.epoch))
    historydf = concat(dflist, axis=1)
    print(historydf)
    metrics_reported = dflist[0].columns
    idx = MultiIndex.from_product(
        [learning_rates, metrics_reported], names=["learning_rate", "metric"]
    )
    historydf.columns = idx
    print(historydf)
    figure(figsize=(12, 8))
    axe = subplot(211)
    historydf.xs("loss", axis=1, level="metric").plot(ylim=(0, 1), ax=axe)
    title("Loss")
    axe = subplot(212)
    historydf.xs("accuracy", axis=1, level="metric").plot(ylim=(0, 1), ax=axe)
    title("Accuracy")
    xlabel("Epochs")
    tight_layout()

    batch(x_train, x_test, y_train, y_test)


def batch(x_train, x_test, y_train, y_test):
    """
    batch sizes function
    """
    dflist = []
    batch_sizes = [16, 32, 64, 128]
    for batch_size in batch_sizes:
        backend.clear_session()
        model = models.Sequential()
        model.add(layers.Dense(1, input_shape=(4,), activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=10)
        dflist.append(DataFrame(history.history, index=history.epoch))
    historydf = concat(dflist, axis=1)
    metrics_reported = dflist[0].columns
    idx = MultiIndex.from_product(
        [batch_sizes, metrics_reported], names=["batch_size", "metric"]
    )
    historydf.columns = idx
    print(historydf)
    figure(figsize=(12, 8))
    axe = subplot(211)
    historydf.xs("loss", axis=1, level="metric").plot(ylim=(0, 1), ax=axe)
    title("Loss")
    axe = subplot(212)
    historydf.xs("accuracy", axis=1, level="metric").plot(ylim=(0, 1), ax=axe)
    title("Accuracy")
    xlabel("Epochs")
    tight_layout()

    optimize(x_train, x_test, y_train, y_test)


def optimize(x_train, x_test, y_train, y_test):
    """
    optimizers function
    """
    dflist = []
    optimizers_list = [
        (optimizers.SGD, {"learning_rate": 0.01}),
        (optimizers.SGD, {"learning_rate": 0.01, "momentum": 0.3}),
        (optimizers.SGD, {"learning_rate": 0.01, "momentum": 0.3, "nesterov": True}),
        (optimizers.Adam, {"learning_rate": 0.01}),
        (optimizers.Adagrad, {"learning_rate": 0.01}),
        (optimizers.RMSprop, {"learning_rate": 0.01}),
    ]
    for opt, param in optimizers_list:
        backend.clear_session()
        model = models.Sequential()
        model.add(layers.Dense(1, input_shape=(4,), activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy",
            optimizer=opt(**param),
            metrics=["accuracy"],
        )
        history = model.fit(x_train, y_train, batch_size=16, epochs=5)
        dflist.append(DataFrame(history.history, index=history.epoch))
    historydf = concat(dflist, axis=1)
    metrics_reported = dflist[0].columns
    optimizers_str = [repr(opt(**param)) for opt, param in optimizers_list]
    idx = MultiIndex.from_product(
        [optimizers_str, metrics_reported], names=["optimizers", "metric"]
    )
    historydf.columns = idx
    figure(figsize=(12, 8))
    axes = subplot(211)
    historydf.xs("loss", axis=1, level="metric").plot(ylim=(0, 1), ax=axes)
    title("Loss")
    axes = subplot(212)
    historydf.xs("accuracy", axis=1, level="metric").plot(ylim=(0, 1), ax=axes)
    title("Accuracy")
    xlabel("Epochs")
    tight_layout()

    initial(x_train, x_test, y_train, y_test)


def initial(x_train, x_test, y_train, y_test):
    """
    initialization function
    """
    dflist = []
    initializers = ["zeros", "uniform", "normal", "he_normal", "lecun_uniform"]
    for init in initializers:
        backend.clear_session()
        model = models.Sequential()
        model.add(
            layers.Dense(
                1, input_shape=(4,), kernel_initializer=init, activation="sigmoid"
            )
        )
        model.compile(
            loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
        )
        history = model.fit(x_train, y_train, batch_size=16, epochs=5)
        dflist.append(DataFrame(history.history, index=history.epoch))
    historydf = concat(dflist, axis=1)
    metrics_reported = dflist[0].columns
    idx = MultiIndex.from_product(
        [initializers, metrics_reported], names=["initializers", "metric"]
    )
    historydf.columns = idx
    figure(figsize=(12, 8))
    axes = subplot(211)
    historydf.xs("loss", axis=1, level="metric").plot(ylim=(0, 1), ax=axes)
    title("Loss")
    axes = subplot(212)
    historydf.xs("accuracy", axis=1, level="metric").plot(ylim=(0, 1), ax=axes)
    title("Accuracy")
    xlabel("Epochs")
    tight_layout()

    inner(x_train, x_test, y_train, y_test)


def inner(x_train, x_test, y_train, y_test):
    """
    inner layer representation
    """
    backend.clear_session()
    model = models.Sequential()
    model.add(layers.Dense(2, input_shape=(4,), activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.RMSprop(learning_rate=0.01),
        metrics=["accuracy"],
    )
    model.fit(x_train, y_train, batch_size=16, epochs=20, validation_split=0.3)
    result = model.evaluate(x_test, y_test)
    print(result)
    model.summary()
    print(model.layers)
    inp = model.layers[0].input
    out = model.layers[0].output
    print(inp)
    print(out)
    features_function = backend.function([inp], [out])
    print(features_function)
    print(features_function([x_test])[0].shape)
    features = features_function([x_test])[0]
    scatter(features[:, 0], features[:, 1], c=y_test, cmap="coolwarm")
    backend.clear_session()
    model = models.Sequential()
    model.add(layers.Dense(3, input_shape=(4,), activation="relu"))
    model.add(layers.Dense(2, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.RMSprop(learning_rate=0.01),
        metrics=["accuracy"],
    )
    inp = model.layers[0].input
    out = model.layers[1].output
    features_function = backend.function([inp], [out])
    figure(figsize=(15, 10))
    for i in range(1, 26):
        subplot(5, 5, i)
        model.fit(x_train, y_train, batch_size=16, epochs=1)
        test_accuracy = model.evaluate(x_test, y_test)[1]
        features = features_function([x_test])[0]
        scatter(features[:, 0], features[:, 1], c=y_test, cmap="coolwarm")
        xlim(-0.5, 3.5)
        ylim(-0.5, 4.0)
        title("Epoch: {}, Test Acc: {:3.1f} %".format(i, test_accuracy * 100.0))
    tight_layout()


if __name__ == "__main__":
    main()
    show()
