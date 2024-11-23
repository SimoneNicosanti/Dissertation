import keras


def buildModel() -> keras.Model:

    # Build model parts
    preprocess = keras.Sequential(
        [
            keras.layers.Rescaling(1.0 / 255),
            keras.layers.Flatten(),
        ],
        name="preprocess",
    )

    firstPart = keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
        ],
        name="first",
    )

    secondPart = keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
        ],
        name="second",
    )

    thirdPart = keras.Sequential(
        [
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
        ],
        name="third",
    )

    forthPart = keras.Sequential(
        [
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
        ],
        name="forth",
    )

    fifthPart = keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
        ],
        name="fifth",
    )

    outputSeq = keras.Sequential(
        [
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ],
        name="out",
    )

    inputs = keras.Input(shape=(28, 28))
    x = preprocess(inputs)
    x_1 = firstPart(x)
    x_2 = secondPart(x)
    x_3 = keras.layers.Add()([x_1, x_2])
    x_4 = thirdPart(x_3)
    x_5 = forthPart(x_4)
    x_6 = fifthPart(x_4)
    x_7 = keras.layers.Add()([x_2, x_5, x_6])

    output = outputSeq(x_7)

    return keras.Model(inputs=inputs, outputs=output)


def main():
    # Load and prepare data
    (trainSet, trainLabels), _ = keras.datasets.mnist.load_data()

    # Full model assembly
    model = buildModel()

    # Compile and train model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    with open("mnist_json.json", "w") as f:
        f.write(model.to_json())
    print(model.summary())

    model.fit(trainSet, trainLabels, epochs=1)

    model.save("../../../models/server/mnist_1.keras")


if __name__ == "__main__":
    main()
