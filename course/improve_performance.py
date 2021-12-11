"""
Improving Performance
"""
from os import getcwd
from os.path import join
from numpy import linspace, array
from tensorflow.keras import (
    models,
    layers,
    backend,
    callbacks,
    utils,
    preprocessing,
    datasets,
)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import (
    show,
    imshow,
    subplot,
    plot,
    legend,
    figure,
    fill_between,
    ylim,
    title,
    xlabel,
    ylabel,
)

path = lambda x: join(getcwd(), "data", x)


def repeated_training(x_train, y_train, x_test, y_test, do_bn=False):
    """
    repeated training function
    """
    units = 512
    activation = "sigmoid"
    optimizer = "sgd"
    epochs = 10
    repeats = 3
    histories = []
    for repeat in range(repeats):
        backend.clear_session()
        model = models.Sequential()
        model.add(
            layers.Dense(
                units,
                input_shape=x_train.shape[1:],
                kernel_initializer="normal",
                activation=activation,
            )
        )
        if do_bn:
            model.add(layers.BatchNormalization())
        model.add(
            layers.Dense(units, kernel_initializer="normal", activation=activation)
        )
        if do_bn:
            model.add(layers.BatchNormalization())
        model.add(
            layers.Dense(units, kernel_initializer="normal", activation=activation)
        )
        if do_bn:
            model.add(layers.BatchNormalization())
        model.add(layers.Dense(10, activation="softmax"))
        model.compile(optimizer, "categorical_crossentropy", metrics=["accuracy"])
        hist = model.fit(
            x_train, y_train, validation_data=(x_test, y_test), epochs=epochs
        )
        histories.append([hist.history["accuracy"], hist.history["val_accuracy"]])
        print(repeat, end=" ")
    histories = array(histories)
    print()

    return (
        histories.mean(axis=0)[0],
        histories.std(axis=0)[0],
        histories.mean(axis=0)[1],
        histories.std(axis=0)[1],
    )


def plot_mean_std(mean, std):
    """
    plot mean std function
    """
    figure()
    plot(mean)
    fill_between(range(len(mean)), mean - std, mean + std, alpha=0.1)


def main():
    """
    main function
    """
    x_data, y_data = load_digits()["data"], load_digits()["target"]
    print(x_data[0])
    print(x_data.shape)
    for i in range(8):
        subplot(1, 8, i + 1)
        imshow(x_data.reshape(-1, 8, 8)[i], cmap="gray")
    backend.clear_session()
    model = models.Sequential()
    model.add(layers.Dense(16, input_shape=(64,), activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, utils.to_categorical(y_data, 10), test_size=0.3
    )
    print(x_train.shape)
    train_sizes = (len(x_train) * linspace(0.1, 0.999, 4)).astype(int)
    print(train_sizes)
    train_scores = []
    test_scores = []
    for train_size in train_sizes:
        x_train_frac, _, y_train_frac, _ = train_test_split(
            x_train, y_train, train_size=train_size
        )
        model.set_weights(model.get_weights())
        model.fit(
            x_train_frac,
            y_train_frac,
            epochs=300,
            callbacks=[callbacks.EarlyStopping(monitor="loss", patience=1)],
        )
        train_scores.append(model.evaluate(x_train_frac, y_train_frac)[-1])
        test_scores.append(model.evaluate(x_test, y_test)[-1])
        print("Done size: ", train_size)
    plot(train_sizes, train_scores, "o-", label="Training score")
    plot(train_sizes, test_scores, "o-", label="Test score")
    legend(loc="best")

    batch(x_train, x_test, y_train, y_test)


def batch(x_train, x_test, y_train, y_test):
    """
    batch normalization function
    """
    mean_acc, std_acc, mean_acc_val, std_acc_val = repeated_training(
        x_train, y_train, x_test, y_test, do_bn=False
    )
    mean_acc_bn, std_acc_bn, mean_acc_val_bn, std_acc_val_bn = repeated_training(
        x_train, y_train, x_test, y_test, do_bn=True
    )
    plot_mean_std(mean_acc, std_acc)
    plot_mean_std(mean_acc_val, std_acc_val)
    plot_mean_std(mean_acc_bn, std_acc_bn)
    plot_mean_std(mean_acc_val_bn, std_acc_val_bn)
    ylim(0, 1.01)
    title("Batch Normalization Accuracy")
    xlabel("Epochs")
    ylabel("Accuracy")
    legend(
        [
            "Train",
            "Test",
            "Train with Batch Normalization",
            "Test with Batch Normalization",
        ],
        loc="best",
    )
    model = models.Sequential()
    model.add(layers.Dropout(0.2, input_shape=x_train.shape[1:]))
    model.add(
        layers.Dense(
            512,
            kernel_initializer="normal",
            kernel_regularizer="l2",
            activation="sigmoid",
        )
    )
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile("sgd", "categorical_crossentropy", metrics=["accuracy"])

    aug()


def aug():
    """
    data augmentation function
    """
    generator = preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=20,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
    )
    train = generator.flow_from_directory(
        path("generator"), target_size=(128, 128), batch_size=32, class_mode="binary"
    )
    figure(figsize=(12, 12))
    for i in range(16):
        img, _ = train.next()
        subplot(4, 4, i + 1)
        imshow(img[0])

    embe()


def embe():
    """
    embeddings function
    """
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=100, output_dim=2))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    emb = model.predict(array([[81, 1, 96, 79], [17, 47, 69, 50], [49, 3, 12, 88]]))
    print(emb.shape)
    print(emb)
    (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(
        "/tmp/imdb.npz",
        num_words=None,
        skip_top=0,
        maxlen=None,
        start_char=1,
        oov_char=2,
        index_from=3,
    )
    print(x_train.shape)
    print(x_train[1])
    idx = datasets.imdb.get_word_index()
    print(max(idx.values()))
    print(idx)
    rev_idx = {v + 3: k for k, v in idx.items()}
    print(rev_idx)
    rev_idx[0] = "padding_char"
    rev_idx[1] = "start_char"
    rev_idx[2] = "oov_char"
    rev_idx[3] = "unk_char"
    print(rev_idx[3])
    print(y_train[0])
    example_review = " ".join([rev_idx[word] for word in x_train[0]])
    print(example_review)
    print(len(x_train[0]))
    print(len(x_train[1]))
    print(len(x_train[2]))
    print(len(x_train[3]))
    maxlen = 100
    x_train_pad = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test_pad = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    print(x_train_pad.shape)
    print(x_train_pad[0])
    print(x_train[0])
    max_features = max([max(x) for x in x_train_pad] + [max(x) for x in x_test_pad]) + 1
    print(max_features)
    print(y_train)

    model = models.Sequential()
    # model.add(layers.Embedding(max_features, 128))
    # model.add(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train_pad, y_train, batch_size=32, epochs=2, validation_split=0.3)
    score, acc = model.evaluate(x_test_pad, y_test)
    print("Test score:", score)
    print("Test accuracy:", acc)


if __name__ == "__main__":
    main()
    show()
