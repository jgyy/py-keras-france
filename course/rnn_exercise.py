"""
Recurrent Neural Networks
"""
from os import getcwd
from os.path import join
from pandas import Timestamp, DataFrame, read_csv, to_datetime
from pandas.tseries.offsets import MonthEnd
from tensorflow.keras import callbacks, models, layers, backend, datasets, utils
from matplotlib.pyplot import show, plot
from sklearn.preprocessing import MinMaxScaler

path = lambda x: join(getcwd(), "data", x)
early_stop = callbacks.EarlyStopping(monitor="loss", patience=1)


def main():
    """
    main function
    """
    daf = DataFrame(
        read_csv(
            path("cansim-0800020-eng-6674700030567901031.csv"),
            skiprows=6,
            skipfooter=9,
            engine="python",
        )
    )
    daf["Adjustments"] = to_datetime(daf["Adjustments"]) + MonthEnd(1)
    daf = daf.set_index("Adjustments")
    print(daf.head())
    split_date = Timestamp("01-01-2011")
    train = daf.loc[:split_date, ["Unadjusted"]]
    test = daf.loc[split_date:, ["Unadjusted"]]
    sca = MinMaxScaler()
    train_sc = sca.fit_transform(train)
    test_sc = sca.transform(test)
    train_sc_df = DataFrame(train_sc, columns=["Scaled"], index=train.index)
    test_sc_df = DataFrame(test_sc, columns=["Scaled"], index=test.index)
    for sca in range(1, 13):
        train_sc_df["shift_{}".format(sca)] = train_sc_df["Scaled"].shift(sca)
        test_sc_df["shift_{}".format(sca)] = test_sc_df["Scaled"].shift(sca)
    x_train = train_sc_df.dropna().drop("Scaled", axis=1)
    y_train = train_sc_df.dropna()[["Scaled"]]
    x_test = test_sc_df.dropna().drop("Scaled", axis=1)
    y_test = test_sc_df.dropna()[["Scaled"]]
    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values
    print(x_train.shape)

    exercise(x_train, x_test, y_train, y_test)


def exercise(x_train, x_test, y_train, y_test):
    """
    exercise 1-2 function
    """
    x_train_t = x_train.reshape(x_train.shape[0], 12, 1)
    x_test_t = x_test.reshape(x_test.shape[0], 12, 1)
    print(x_train_t.shape)
    backend.clear_session()
    model = models.Sequential()
    model.add(layers.LSTM(6, input_shape=(12, 1)))
    model.add(layers.Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.summary()
    model.fit(x_train_t, y_train, epochs=600, batch_size=32)
    y_pred = model.predict(x_test_t)
    plot(y_test)
    plot(y_pred)
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train_cat = utils.to_categorical(y_train, 10)
    y_test_cat = utils.to_categorical(y_test, 10)
    x_train = x_train.reshape(x_train.shape[0], -1, 1)
    x_test = x_test.reshape(x_test.shape[0], -1, 1)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train_cat.shape)
    print(y_test_cat.shape)
    backend.clear_session()
    model = models.Sequential()
    model.add(layers.LSTM(32, input_shape=x_train.shape[1:]))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )
    model.fit(
        x_train,
        y_train_cat,
        batch_size=32,
        epochs=100,
        validation_split=0.3,
        shuffle=True,
    )
    model.evaluate(x_test, y_test_cat)


if __name__ == "__main__":
    main()
    show()
