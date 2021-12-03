"""
Data Exploration Exercises
"""
from os import getcwd
from os.path import join
from pandas import DataFrame, read_csv, to_datetime
from pandas.plotting import scatter_matrix
from matplotlib.pyplot import (
    figure,
    show,
    subplots,
    xlabel,
    ylabel,
    title,
    legend,
    axvline,
    axhline,
)

path = lambda x: join(getcwd(), "data", x)


def main():
    """
    main function
    """
    dataf = DataFrame(read_csv(path("international-airline-passengers.csv")))
    print(dataf.head())
    dataf["Month"] = to_datetime(dataf["Month"])
    dataf = dataf.set_index("Month")
    print(dataf.head())
    dataf.plot()
    dataf = DataFrame(read_csv(path("weight-height.csv")))
    print(dataf.head())
    print(dataf.describe())
    print(dataf["Gender"].value_counts())
    dataf.plot(kind="scatter", x="Height", y="Weight")
    males = dataf[dataf["Gender"] == "Male"]
    females = dataf.query('Gender == "Female"')
    _, axs = subplots()
    males.plot(
        kind="scatter",
        x="Height",
        y="Weight",
        ax=axs,
        color="blue",
        alpha=0.3,
        title="Male & Female Populations",
    )
    females.plot(kind="scatter", x="Height", y="Weight", ax=axs, color="red", alpha=0.3)
    dataf["Gendercolor"] = dataf["Gender"].map({"Male": "blue", "Female": "red"})
    print(dataf.head())
    dataf.plot(
        kind="scatter",
        x="Height",
        y="Weight",
        c=dataf["Gendercolor"],
        alpha=0.3,
        title="Male & Female Populations",
    )
    _, axs = subplots()
    axs.plot(
        males["Height"],
        males["Weight"],
        "ob",
        females["Height"],
        females["Weight"],
        "or",
        alpha=0.3,
    )
    xlabel("Height")
    ylabel("Weight")
    title("Male & Female Populations")
    figure()
    males["Height"].plot(kind="hist", bins=50, range=(50, 80), alpha=0.3, color="blue")
    females["Height"].plot(kind="hist", bins=50, range=(50, 80), alpha=0.3, color="red")
    title("Height distribution")
    legend(["Males", "Females"])
    xlabel("Heigth (in)")
    axvline(males["Height"].mean(), color="blue", linewidth=2)
    axvline(females["Height"].mean(), color="red", linewidth=2)
    figure()
    males["Height"].plot(
        kind="hist",
        bins=200,
        range=(50, 80),
        alpha=0.3,
        color="blue",
        cumulative=True,
        density=True,
    )
    females["Height"].plot(
        kind="hist",
        bins=200,
        range=(50, 80),
        alpha=0.3,
        color="red",
        cumulative=True,
        density=True,
    )
    title("Height distribution")
    legend(["Males", "Females"])
    xlabel("Heigth (in)")
    axhline(0.8)
    axhline(0.5)
    axhline(0.2)
    dfpvt = dataf.pivot(columns="Gender", values="Weight")
    print(dfpvt.head())
    dfpvt.plot(kind='box')
    title('Weight Box Plot')
    ylabel("Weight (lbs)")
    dataf = DataFrame(read_csv(path('titanic-train.csv')))
    print(dataf.head())
    scatter_matrix(dataf.drop('PassengerId', axis=1), figsize=(10, 10))


if __name__ == "__main__":
    main()
    show()
