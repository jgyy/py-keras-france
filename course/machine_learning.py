"""
Linear Regression
"""
from os import getcwd
from os.path import join
from pandas import DataFrame, read_csv, get_dummies
from numpy import linspace, array, zeros
from matplotlib.pyplot import show, plot, figure, subplot, title, xlabel, legend
from tensorflow.keras import models, layers, optimizers, wrappers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error as mse,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

path = lambda x: join(getcwd(), "data", x)
line = lambda x, w=0, b=0: x * w + b
mean_squared_error = lambda y_true, y_pred: ((y_true - y_pred) ** 2).mean()


def build_logistic_regression_model():
    """
    build logistic regression model function
    """
    model = models.Sequential()
    model.add(layers.Dense(1, input_shape=(1,), activation="sigmoid"))
    model.compile(
        optimizers.SGD(learning_rate=0.5),
        "binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def pretty_confusion_matrix(y_true, y_pred, labels=None):
    """
    pretty confusion matrix function
    """
    if not labels:
        labels = ["False", "True"]
    confusem = confusion_matrix(y_true, y_pred)
    pred_labels = ["Predicted " + l for l in labels]
    dataf = DataFrame(confusem, index=labels, columns=pred_labels)
    return dataf


def main():
    """
    main function
    """
    dataf = DataFrame(read_csv(path("weight-height.csv")))
    print(dataf.head())
    dataf.plot(
        kind="scatter", x="Height", y="Weight", title="Weight and Height in adults"
    )
    dataf.plot(
        kind="scatter", x="Height", y="Weight", title="Weight and Height in adults"
    )
    plot([55, 78], [75, 250], color="red", linewidth=3)
    x_array = linspace(55, 80, 100)
    print(x_array)
    yhat = line(x_array, 0, 0)
    print(yhat)
    dataf.plot(
        kind="scatter", x="Height", y="Weight", title="Weight and Height in adults"
    )
    plot(x_array, yhat, color="red", linewidth=3)

    cost_function(dataf)


def cost_function(dataf):
    """
    cost function
    """
    x_height = dataf[["Height"]].values
    y_true = dataf["Weight"].values
    print(y_true)
    y_pred = line(x_height)
    print(y_pred)
    print(mean_squared_error(y_true, y_pred.ravel()))
    figure(figsize=(10, 5))
    ax1 = subplot(121)
    dataf.plot(
        kind="scatter",
        x="Height",
        y="Weight",
        title="Weight and Height in adults",
        ax=ax1,
    )
    bbs = array([-100, -50, 0, 50, 100, 150])
    mses = []
    for b_int in bbs:
        y_pred = line(x_height, 2, b_int)
        mserror = mean_squared_error(y_true, y_pred)
        mses.append(mserror)
        plot(x_height, y_pred)
    subplot(122)
    plot(bbs, mses, "o-")
    title("Cost as a function of b")
    xlabel("b")

    keras(dataf, x_height, y_true)


def keras(dataf, x_height, y_true):
    """
    Linear Regression with Keras
    """
    model = models.Sequential()
    model.add(layers.Dense(1, input_shape=(1,)))
    model.summary()
    model.compile(optimizers.Adam(learning_rate=0.8), "mean_squared_error")
    model.fit(x_height, y_true, epochs=40)
    y_pred = model.predict(x_height)
    dataf.plot(
        kind="scatter", x="Height", y="Weight", title="Weight and Height in adults"
    )
    plot(x_height, y_pred, color="red")
    w_array, b_array = model.get_weights()
    print(w_array)
    print(b_array)
    print("The R2 score is {:0.3f}".format(r2_score(y_true, y_pred)))
    x_train, x_test, y_train, y_test = train_test_split(x_height, y_true, test_size=0.2)
    print(len(x_train))
    print(len(x_test))
    w_array[0, 0] = 0.0
    b_array[0] = 0.0
    model.set_weights((w_array, b_array))
    model.fit(x_train, y_train, epochs=50, verbose=0)
    y_train_pred = model.predict(x_train).ravel()
    y_test_pred = model.predict(x_test).ravel()
    print(
        f"The Mean Squared Error on the Train set is:\t{mse(y_train, y_train_pred):0.1f}"
    )
    print(
        f"The Mean Squared Error on the Test set is:\t{mse(y_test, y_test_pred):0.1f}"
    )
    print(f"The R2 score on the Train set is:\t{r2_score(y_train, y_train_pred):0.3f}")
    print(f"The R2 score on the Test set is:\t{r2_score(y_test, y_test_pred):0.3f}")

    classification()


def classification():
    """
    classification function
    """
    dataf = DataFrame(read_csv(path("user_visit_duration.csv")))
    print(dataf.head())
    dataf.plot(kind="scatter", x="Time (min)", y="Buy")
    model = models.Sequential()
    model.add(layers.Dense(1, input_shape=(1,), activation="sigmoid"))
    model.compile(
        optimizers.SGD(learning_rate=0.5), "binary_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    x_time = dataf[["Time (min)"]].values
    y_buy = dataf["Buy"].values
    model.fit(x_time, y_buy, epochs=25)
    axs = dataf.plot(
        kind="scatter",
        x="Time (min)",
        y="Buy",
        title="Purchase behavior VS time spent on site",
    )
    temp = linspace(0, 4)
    axs.plot(temp, model.predict(temp), color="orange")
    legend(["model", "data"])
    temp_class = model.predict(temp) > 0.5
    axs = dataf.plot(
        kind="scatter",
        x="Time (min)",
        y="Buy",
        title="Purchase behavior VS time spent on site",
    )
    temp = linspace(0, 4)
    axs.plot(temp, temp_class, color="orange")
    legend(["model", "data"])
    y_pred = model.predict(x_time)
    y_class_pred = y_pred > 0.5
    print("The accuracy score is {:0.3f}".format(accuracy_score(y_buy, y_class_pred)))

    tts(model, x_time, y_buy, y_class_pred)


def tts(model, x_time, y_buy, y_class_pred):
    """
    train test split function
    """
    x_train, x_test, y_train, y_test = train_test_split(x_time, y_buy, test_size=0.2)
    params = model.get_weights()
    params = [zeros(w.shape) for w in params]
    model.set_weights(params)
    print(
        f"The accuracy score is {accuracy_score(y_buy, model.predict(x_time) > 0.5):0.3f}"
    )
    model.fit(x_train, y_train, epochs=25, verbose=0)
    print(
        f"The train accuracy score is {accuracy_score(y_train, model.predict(x_train) > 0.5):0.3f}"
    )
    print(
        f"The test accuracy score is {accuracy_score(y_test, model.predict(x_test) > 0.5):0.3f}"
    )
    model = wrappers.scikit_learn.KerasClassifier(
        build_fn=build_logistic_regression_model, epochs=25, verbose=0
    )
    crossv = KFold(3, shuffle=True)
    scores = cross_val_score(model, x_time, y_buy, cv=crossv)
    print(scores)
    print(
        f"The cross validation accuracy is {scores.mean():0.4f} ± {scores.std():0.4f}"
    )
    print(confusion_matrix(y_buy, y_class_pred))
    print(pretty_confusion_matrix(y_buy, y_class_pred, ["Not Buy", "Buy"]))
    print("Precision:\t{:0.3f}".format(precision_score(y_buy, y_class_pred)))
    print("Recall:  \t{:0.3f}".format(recall_score(y_buy, y_class_pred)))
    print("F1 Score:\t{:0.3f}".format(f1_score(y_buy, y_class_pred)))
    print(classification_report(y_buy, y_class_pred))
    dataf = DataFrame(read_csv(path("weight-height.csv")))
    print(dataf.head())
    print(dataf["Gender"].unique())
    print(get_dummies(dataf["Gender"], prefix="Gender").head())
    dataf["Height (feet)"] = dataf["Height"] / 12.0
    dataf["Weight (100 lbs)"] = dataf["Weight"] / 100.0
    print(dataf.describe().round(2))
    dataf["Weight_mms"] = MinMaxScaler().fit_transform(dataf[["Weight"]])
    dataf["Height_mms"] = MinMaxScaler().fit_transform(dataf[["Height"]])
    print(dataf.describe().round(2))
    dataf["Weight_ss"] = StandardScaler().fit_transform(dataf[["Weight"]])
    dataf["Height_ss"] = StandardScaler().fit_transform(dataf[["Height"]])
    print(dataf.describe().round(2))
    figure(figsize=(15, 5))
    for i, feature in enumerate(["Height", "Height (feet)", "Height_mms", "Height_ss"]):
        subplot(1, 4, i + 1)
        dataf[feature].plot(kind="hist", title=feature)
        xlabel(feature)


if __name__ == "__main__":
    main()
    show()
