"""
Linear Regression
"""
from os import getcwd
from os.path import join
from numpy import argmax
from pandas import DataFrame, read_csv, get_dummies, concat
from matplotlib.pyplot import show, figure, subplot, xlabel
from sklearn.metrics import r2_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from tensorflow.keras import models, layers, optimizers
from scikeras.wrappers import KerasClassifier

path = lambda x: join(getcwd(), "data", x)


def pretty_confusion_matrix(y_true, y_pred, labels=None):
    """
    pretty confusion matrix function
    """
    if not labels:
        labels = ["False", "True"]
    confusionm = confusion_matrix(y_true, y_pred)
    pred_labels = ["Predicted " + l for l in labels]
    dataf = DataFrame(confusionm, index=labels, columns=pred_labels)
    return dataf


def build_logistic_regression_model():
    """
    build logistic regression model function
    """
    model = models.Sequential()
    model.add(layers.Dense(1, input_dim=20, activation="sigmoid"))
    model.compile(
        optimizers.Adam(learning_rate=0.5), "binary_crossentropy", metrics=["accuracy"]
    )
    return model


def main():
    """
    main function
    """
    dataf = DataFrame(read_csv(path("housing-data.csv")))
    print(dataf.head())
    print(dataf.columns)
    figure(figsize=(15, 5))
    for i, feature in enumerate(dataf.columns):
        subplot(1, 4, i + 1)
        dataf[feature].plot(kind="hist", title=feature)
        xlabel(feature)
    x_data = dataf[["sqft", "bdrms", "age"]].values
    y_data = dataf["price"].values
    print(x_data)
    print(y_data)
    model = models.Sequential()
    model.add(layers.Dense(1, input_shape=(3,)))
    model.compile(optimizers.Adam(learning_rate=0.8), "mean_squared_error")
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    print(len(x_train))
    print(len(x_data))
    model.fit(x_train, y_train, epochs=10)
    print(dataf.describe())
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print(f"The R2 score on the Train set is:\t{r2_score(y_train, y_train_pred):0.3f}")
    print(f"The R2 score on the Test set is:\t{r2_score(y_test, y_test_pred):0.3f}")
    dataf["sqft1000"] = dataf["sqft"] / 1000.0
    dataf["age10"] = dataf["age"] / 10.0
    dataf["price100k"] = dataf["price"] / 1e5
    x_data = dataf[["sqft1000", "bdrms", "age10"]].values
    y_data = dataf["price100k"].values
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    model = models.Sequential()
    model.add(layers.Dense(1, input_dim=3))
    model.compile(optimizers.Adam(learning_rate=0.1), "mean_squared_error")
    model.fit(x_train, y_train, epochs=20)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print(f"The R2 score on the Train set is:\t{r2_score(y_train, y_train_pred):0.3f}")
    print(f"The R2 score on the Test set is:\t{r2_score(y_test, y_test_pred):0.3f}")
    model.fit(x_train, y_train, epochs=40)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print(f"The R2 score on the Train set is:\t{r2_score(y_train, y_train_pred):0.3f}")
    print(f"The R2 score on the Test set is:\t{r2_score(y_test, y_test_pred):0.3f}")


def exercise():
    """
    exercise 2 function
    """
    dataf = DataFrame(read_csv(path("HR_comma_sep.csv")))
    print(dataf.head())
    print(dataf.info())
    print(dataf.describe())
    print(dataf.left.value_counts() / len(dataf))
    dataf["average_montly_hours"].plot(kind="hist")
    dataf["average_montly_hours_100"] = dataf["average_montly_hours"] / 100.0
    dataf["average_montly_hours_100"].plot(kind="hist")
    dataf["time_spend_company"].plot(kind="hist")
    df_dummies = get_dummies(dataf[["sales", "salary"]])
    print(df_dummies.head())
    print(dataf.columns)
    x_data = concat(
        [
            dataf[
                [
                    "satisfaction_level",
                    "last_evaluation",
                    "number_project",
                    "time_spend_company",
                    "Work_accident",
                    "promotion_last_5years",
                    "average_montly_hours_100",
                ]
            ],
            df_dummies,
        ],
        axis=1,
    ).values
    y_data = dataf["left"].values
    print(x_data.shape)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    model = models.Sequential()
    model.add(layers.Dense(1, input_dim=20, activation="sigmoid"))
    model.compile(
        optimizers.Adam(learning_rate=0.5), "binary_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    model.fit(x_train, y_train, epochs=10)
    y_test_pred = argmax(model.predict(x_test), axis=-1)
    pretty_confusion_matrix(y_test, y_test_pred, labels=["Stay", "Leave"])
    print(classification_report(y_test, y_test_pred, zero_division=1))
    model = KerasClassifier(model=build_logistic_regression_model, epochs=10, verbose=0)
    crossv = KFold(5, shuffle=True)
    scores = cross_val_score(model, x_data, y_data, cv=crossv)
    print(
        f"The cross validation accuracy is {scores.mean():0.4f} ± {scores.std():0.4f}"
    )
    print(scores)


if __name__ == "__main__":
    main()
    exercise()
    show()
