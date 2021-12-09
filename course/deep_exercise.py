"""
Deep Learning Intro
"""
from os import getcwd
from os.path import join
from numpy import argmax
from pandas import DataFrame, Series, read_csv
from seaborn import pairplot, heatmap
from matplotlib.pyplot import show
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras import models, layers, optimizers, utils

path = lambda x: join(getcwd(), "data", x)


def main():
    """
    main function
    """
    dataf = DataFrame(read_csv(path("diabetes.csv")))
    print(dataf.head())
    dataf.hist(figsize=(12, 10))
    pairplot(dataf, hue="Outcome")
    heatmap(dataf.corr(), annot=True)
    print(dataf.info())
    print(dataf.describe())
    sca = StandardScaler()
    x_data = sca.fit_transform(dataf.drop("Outcome", axis=1))
    y_data = dataf["Outcome"].values
    y_cat = utils.to_categorical(y_data)
    print(x_data.shape)
    print(y_cat.shape)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_cat, random_state=22, test_size=0.2
    )
    model = models.Sequential()
    model.add(layers.Dense(32, input_shape=(8,), activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(
        optimizers.Adam(learning_rate=0.05),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    print(32 * 8 + 32)
    model.fit(x_train, y_train, epochs=20, validation_split=0.1)
    y_pred = model.predict(x_test)
    y_test_class = argmax(y_test, axis=1)
    y_pred_class = argmax(y_pred, axis=1)
    print(Series(y_test_class).value_counts() / len(y_test_class))
    print(accuracy_score(y_test_class, y_pred_class))
    print(classification_report(y_test_class, y_pred_class))
    print(confusion_matrix(y_test_class, y_pred_class))

    for mod in [RandomForestClassifier(), SVC(), GaussianNB()]:
        mod.fit(x_train, y_train[:, 1])
        y_pred = mod.predict(x_test)
        print("=" * 80)
        print(mod)
        print("-" * 80)
        print("Accuracy score: {:0.3}".format(accuracy_score(y_test_class, y_pred)))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test_class, y_pred))
        print()


if __name__ == "__main__":
    main()
    show()
