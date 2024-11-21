import keras
import tensorflow as tf


def main():
    # Load and prepare data
    (trainSet, trainLabels), _ = keras.datasets.mnist.load_data()

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
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
        ],
        name="first",
    )

    secondPart = keras.Sequential(
        [
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ],
        name="second",
    )

    # Full model assembly
    model = keras.Sequential(
        [keras.Input(shape=(28, 28)), preprocess, firstPart, secondPart], name="full"
    )

    # Compile and train model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(trainSet, trainLabels, epochs=2)

    signatures = {
        "full": tf.TensorSpec(shape=(None, 28, 28), dtype=tf.float32, name="x"),
        "preprocess": tf.TensorSpec(shape=(None, 28, 28), dtype=tf.float32, name="x"),
        "first": tf.TensorSpec(shape=(None, 28 * 28), dtype=tf.float32, name="x"),
        "second": tf.TensorSpec(shape=(None, 64), dtype=tf.float32, name="x"),
    }

    ## Si possono esportare:
    ##  - Modello singolo con tutti i pesi e più signature
    ##      >>> Meglio: più ordinato visto che il dispositivo comunque il modello lo deve avere tutto
    ##  - Singole parti con pesi solo della parte e unica signature
    modelParts = [model, preprocess, firstPart, secondPart]

    expArch = keras.export.ExportArchive()
    expArch.track(model)
    for modelPart in modelParts:
        expArch.add_endpoint(
            name=modelPart.name,
            fn=modelPart.__call__,
            input_signature=[signatures[modelPart.name]],
        )
    expArch.write_out("../../../models/server/mnist/1")
    model.save("../../../models/server/mnist.keras")

    converter = tf.lite.TFLiteConverter.from_saved_model(
        "../../../models/server/mnist/1"
    )
    convertedModel = converter.convert()
    with open("../../../models/client/mnist.tflite", "wb") as f:
        f.write(convertedModel)


if __name__ == "__main__":
    main()
