"""
Data Exploration with Pandas
"""
from os import getcwd
from os.path import join
from pandas import DataFrame, read_csv, merge
from numpy import vstack, linspace, asarray, sin
from numpy.random import normal, random
from PIL import Image
from scipy.io import wavfile
from simpleaudio import WaveObject
from matplotlib.pyplot import (
    figure,
    show,
    plot,
    title,
    legend,
    subplots,
    tight_layout,
    specgram,
    ylabel,
    xlabel,
)

path = lambda x: join(getcwd(), "data", x)


def main():
    """
    main function
    """
    dataf = DataFrame(read_csv(path("titanic-train.csv")))
    print(type(dataf))
    print(dataf.head())
    print(dataf.info())
    print(dataf.describe())
    print(dataf.iloc[3])
    print(dataf.loc[0:4, "Ticket"])
    print(dataf["Ticket"].head())
    print(dataf[["Embarked", "Ticket"]].head())
    print(dataf[dataf["Age"] > 70])
    print(dataf["Age"] > 70)
    print(dataf.query("Age > 70"))
    print(dataf[(dataf["Age"] == 11) & (dataf["SibSp"] == 5)])
    print(dataf[(dataf.Age == 11) | (dataf.SibSp == 5)])
    print(dataf.query("(Age == 11) | (SibSp == 5)"))
    print(dataf["Embarked"].unique())
    print(dataf.sort_values("Age", ascending=False).head())
    print(dataf["Survived"].value_counts())
    print(dataf["Pclass"].value_counts())
    print(dataf.groupby(["Pclass", "Survived"])["PassengerId"].count())
    print(dataf["Age"].min())
    print(dataf["Age"].max())
    print(dataf["Age"].mean())
    print(dataf["Age"].median())
    mean_age_by_survived = dataf.groupby("Survived")["Age"].mean()
    print(mean_age_by_survived)
    std_age_by_survived = dataf.groupby("Survived")["Age"].std()
    print(std_age_by_survived)
    df1 = mean_age_by_survived.round(0).reset_index()
    df2 = std_age_by_survived.round(0).reset_index()
    print(df1)
    print(df2)
    df3 = merge(df1, df2, on="Survived")
    print(df3)
    df3.columns = ["Survived", "Average Age", "Age Standard Deviation"]
    print(df3)
    dataf.pivot_table(
        index="Pclass", columns="Survived", values="PassengerId", aggfunc="count"
    )
    dataf["IsFemale"] = dataf["Sex"] == "female"
    correlated_with_survived = dataf.corr()["Survived"].sort_values()
    print(correlated_with_survived)
    figure()
    correlated_with_survived.iloc[:-1].plot(
        kind="bar", title="Titanic Passengers: correlation with survival"
    )


def visual():
    """
    Visual Data Exploration with Matplotlib
    """
    data1 = normal(0, 0.1, 1000)
    data2 = normal(1, 0.4, 1000) + linspace(0, 1, 1000)
    data3 = 2 + random(1000) * linspace(1, 5, 1000)
    data4 = normal(3, 0.2, 1000) + 0.3 * sin(linspace(0, 20, 1000))
    data = vstack([data1, data2, data3, data4]).transpose()
    dataf = DataFrame(data, columns=["data1", "data2", "data3", "data4"])
    print(dataf.head())
    dataf.plot(title="Line plot")
    plot(dataf)
    title("Line plot")
    legend(["data1", "data2", "data3", "data4"])
    dataf.plot(style=".")
    dataf.plot(kind="scatter", x="data1", y="data2", xlim=(-1.5, 1.5), ylim=(0, 3))
    dataf.plot(kind="hist", bins=50, title="Histogram", alpha=0.6)
    dataf.plot(
        kind="hist",
        bins=100,
        title="Cumulative distributions",
        density=True,
        cumulative=True,
        alpha=0.4,
    )
    _, axs = subplots(2, 2, figsize=(5, 5))
    dataf.plot(ax=axs[0][0], title="Line plot")
    dataf.plot(ax=axs[0][1], style="o", title="Scatter plot")
    dataf.plot(ax=axs[1][0], kind="hist", bins=50, title="Histogram")
    dataf.plot(ax=axs[1][1], kind="box", title="Boxplot")
    tight_layout()
    gt01 = dataf["data1"] > 0.1
    piecounts = gt01.value_counts()
    print(piecounts)
    piecounts.plot(
        kind="pie",
        figsize=(5, 5),
        explode=[0, 0.15],
        labels=["<= 0.1", "> 0.1"],
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
        fontsize=16,
    )
    data = vstack(
        [normal((0, 0), 2, size=(1000, 2)), normal((9, 9), 3, size=(2000, 2))]
    )
    dataf = DataFrame(data, columns=["x", "y"])
    print(dataf.head())
    dataf.plot()
    dataf.plot(kind="kde")
    dataf.plot(kind="hexbin", x="x", y="y", bins=100, cmap="rainbow")
    img = Image.open(path("iss.jpg"))
    img.show()
    print(type(img))
    imgarray = asarray(img)
    print(type(imgarray))
    print(imgarray.shape)
    print(imgarray.ravel().shape)
    print(435 * 640 * 3)
    _, snd = wavfile.read(filename=path("sms.wav"))
    wave_obj = WaveObject.from_wave_file(path("sms.wav"))
    wave_obj.play()
    print(len(snd))
    print(snd)
    plot(snd)
    specgram(snd, NFFT=1024, Fs=44100)
    ylabel("Frequency (Hz)")
    xlabel("Time (s)")


if __name__ == "__main__":
    main()
    visual()
    show()
