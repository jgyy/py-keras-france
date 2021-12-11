"""
Recurrent Neural Networks
"""
from os import getcwd
from os.path import join
from numpy import reshape
from pandas import Timestamp, DataFrame, read_csv, to_datetime
from pandas.tseries.offsets import MonthEnd
from tensorflow.keras import callbacks, models, layers, backend
from matplotlib.pyplot import show, legend, plot
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
    print(daf.head())
    daf["Adjustments"] = to_datetime(daf["Adjustments"]) + MonthEnd(1)
    daf = daf.set_index("Adjustments")
    print(daf.head())
    daf.plot()
    split_date = Timestamp("01-01-2011")
    train = daf.loc[:split_date, ["Unadjusted"]]
    test = daf.loc[split_date:, ["Unadjusted"]]
    axe = train.plot()
    test.plot(ax=axe)
    legend(["train", "test"])
    sca = MinMaxScaler()
    train_sc = sca.fit_transform(train)
    test_sc = sca.transform(test)
    print(train_sc[:4])

    predictor(train_sc, test_sc, train, test)


def predictor(train_sc, test_sc, train, test):
    """
    fully connected predictor function
    """
    x_train = train_sc[:-1]
    y_train = train_sc[1:]
    x_test = test_sc[:-1]
    y_test = test_sc[1:]
    model = models.Sequential()
    model.add(layers.Dense(12, input_dim=1, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.summary()
    model.fit(x_train, y_train, epochs=200, batch_size=2, callbacks=[early_stop])
    y_pred = model.predict(x_test)
    plot(y_test)
    plot(y_pred)

    recurrent(train_sc, test_sc, train, test)


def recurrent(train_sc, test_sc, train, test):
    """
    recurrent predictor
    """
    x_train = train_sc[:-1]
    y_train = train_sc[1:]
    x_test = test_sc[:-1]
    y_test = test_sc[1:]
    print(x_train.shape)
    print(x_train[:, None].shape)
    x_train_t = x_train[:, None]
    x_test_t = x_test[:, None]
    backend.clear_session()
    model = models.Sequential()
    # model.add(layers.LSTM(6, input_shape=(1, 1)))
    model.add(layers.Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(x_train_t, y_train, epochs=100, batch_size=1, callbacks=[early_stop])
    y_pred = model.predict(x_test_t)
    plot(y_test)
    # plot(y_pred)
    plot(reshape(y_pred, y_test.shape))

    windows(train_sc, test_sc, train, test)


def windows(train_sc, test_sc, train, test):
    """
    Windows Function
    """
    print(train_sc.shape)
    train_sc_df = DataFrame(train_sc, columns=["Scaled"], index=train.index)
    test_sc_df = DataFrame(test_sc, columns=["Scaled"], index=test.index)
    print(train_sc_df.head())
    for sca in range(1, 13):
        train_sc_df["shift_{}".format(sca)] = train_sc_df["Scaled"].shift(sca)
        test_sc_df["shift_{}".format(sca)] = test_sc_df["Scaled"].shift(sca)
    print(train_sc_df.head(13))
    x_train = train_sc_df.dropna().drop("Scaled", axis=1)
    y_train = train_sc_df.dropna()[["Scaled"]]
    x_test = test_sc_df.dropna().drop("Scaled", axis=1)
    y_test = test_sc_df.dropna()[["Scaled"]]
    print(x_train.head())
    print(x_train.shape)
    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values
    backend.clear_session()
    model = models.Sequential()
    model.add(layers.Dense(12, input_dim=12, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.summary()
    model.fit(x_train, y_train, epochs=200, batch_size=1, callbacks=[early_stop])
    y_pred = model.predict(x_test)
    plot(y_test)
    plot(y_pred)
    x_train_t = x_train.reshape(x_train.shape[0], 1, 12)
    x_test_t = x_test.reshape(x_test.shape[0], 1, 12)
    print(x_train_t.shape)
    backend.clear_session()
    model = models.Sequential()

    # model.add(layers.LSTM(6, input_shape=(1, 12)))
    model.add(layers.Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.build(x_train_t.shape)
    model.summary()
    model.fit(x_train_t, y_train, epochs=100, batch_size=1, callbacks=[early_stop])
    y_pred = model.predict(x_test_t)
    plot(y_test)
    # plot(y_pred)
    plot(reshape(y_pred, y_test.shape))


if __name__ == "__main__":
    main()
    show()
