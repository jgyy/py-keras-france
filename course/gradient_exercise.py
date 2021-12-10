"""
Gradient Descent Exercise
"""
from os import getcwd, system
from os.path import join
from pandas import DataFrame, read_csv, get_dummies
from seaborn import pairplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers, optimizers, backend, callbacks
from matplotlib.pyplot import show, scatter

path = lambda x: join(getcwd(), "data", x)


def main():
    """
    main function
    """
    dataf = DataFrame(read_csv(path("wines.csv")))
    print(dataf.head())
    y_data = dataf["Class"]
    print(y_data.value_counts())
    y_cat = get_dummies(y_data)
    print(y_cat.head())
    x_data = dataf.drop("Class", axis=1)
    print(x_data.shape)
    pairplot(dataf, hue="Class")
    scaler = StandardScaler()
    xsc = scaler.fit_transform(x_data)
    backend.clear_session()
    model = models.Sequential()
    model.add(
        layers.Dense(
            5, input_shape=(13,), kernel_initializer="he_normal", activation="relu"
        )
    )
    model.add(layers.Dense(3, activation="softmax"))
    model.compile(
        optimizers.RMSprop(learning_rate=0.1),
        "categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(xsc, y_cat.values, batch_size=8, epochs=10, validation_split=0.2)

    exercise(xsc, y_cat, y_data)


def exercise(xsc, y_cat, y_data):
    """
    exercise 2-4 function
    """
    backend.clear_session()
    model = models.Sequential()
    model.add(
        layers.Dense(
            8, input_shape=(13,), kernel_initializer="he_normal", activation="tanh"
        )
    )
    model.add(layers.Dense(5, kernel_initializer="he_normal", activation="tanh"))
    model.add(layers.Dense(2, kernel_initializer="he_normal", activation="tanh"))
    model.add(layers.Dense(3, activation="softmax"))
    model.compile(
        optimizers.RMSprop(learning_rate=0.05),
        "categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(xsc, y_cat.values, batch_size=16, epochs=20)
    model.summary()
    inp = model.layers[0].input
    out = model.layers[2].output
    features_function = backend.function([inp], [out])
    features = features_function([xsc])[0]
    print(features.shape)
    scatter(features[:, 0], features[:, 1], c=y_data)
    backend.clear_session()
    inputs = layers.Input(shape=(13,))
    x_input = layers.Dense(8, kernel_initializer="he_normal", activation="tanh")(inputs)
    x_input = layers.Dense(5, kernel_initializer="he_normal", activation="tanh")(
        x_input
    )
    second_to_last = layers.Dense(2, kernel_initializer="he_normal", activation="tanh")(
        x_input
    )
    outputs = layers.Dense(3, activation="softmax")(second_to_last)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizers.RMSprop(learning_rate=0.05),
        "categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(xsc, y_cat.values, batch_size=16, epochs=20)
    features_function = backend.function([inputs], [second_to_last])
    features = features_function([xsc])[0]
    scatter(features[:, 0], features[:, 1], c=y_data)
    checkpointer = callbacks.ModelCheckpoint(
        filepath="/tmp/udemy/weights.hdf5", save_best_only=True
    )
    earlystopper = callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=1, mode="auto"
    )
    tensorboard = callbacks.TensorBoard(log_dir="/tmp/udemy/tensorboard/")
    x_train, x_test, y_train, y_test = train_test_split(
        xsc, y_cat.values, test_size=0.3
    )
    backend.clear_session()
    inputs = layers.Input(shape=(13,))
    x_layer = layers.Dense(8, kernel_initializer="he_normal", activation="tanh")(inputs)
    x_layer = layers.Dense(5, kernel_initializer="he_normal", activation="tanh")(
        x_layer
    )
    second_to_last = layers.Dense(2, kernel_initializer="he_normal", activation="tanh")(
        x_layer
    )
    outputs = layers.Dense(3, activation="softmax")(second_to_last)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizers.RMSprop(learning_rate=0.05),
        "categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=20,
        validation_data=(x_test, y_test),
        callbacks=[checkpointer, earlystopper, tensorboard],
    )


if __name__ == "__main__":
    main()
    show()
    system("tensorboard --logdir /tmp/udemy/tensorboard/")
