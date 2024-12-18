import keras
import numpy as np
import tensorflow as tf
from Manipulation import Split, Unnest


def subModel_1():
    inp_1 = keras.layers.Input(shape=(32,))
    inp_2 = keras.layers.Input(shape=(32,))
    x1 = keras.layers.Dense(units=32)(inp_1)
    x2 = keras.layers.Dense(units=32)(inp_2)
    x3 = x1 + x2
    return keras.Model(inputs=[inp_1, inp_2], outputs=x3)


def subModel():
    inp = keras.layers.Input(shape=(32,))
    x1 = keras.layers.Dense(units=32)(inp)
    x2 = keras.layers.Dense(units=32)(inp)
    x3 = subModel_1()([x1, x2])
    x4 = keras.layers.Dense(units=32)(x3)
    mod_1 = keras.Model(inputs=inp, outputs=[x1, x2, x4])
    return mod_1


def main_1():
    model = keras.saving.load_model("./models/UnnestedYolo.keras")

    subModels: list[keras.Model] = Split.modelSplit(model, maxLayerNum=50)
    for idx, subMod in enumerate(subModels):
        subMod.save(f"./models/SubYolo_{idx}.keras")


def main_2():
    inp_1 = keras.Input(shape=(32,))
    x = subModel()(inp_1)
    x0 = keras.layers.Dense(units=1)(x[0])
    x1 = keras.layers.Dense(units=1)(x[1])
    x2 = keras.layers.Dense(units=1)(x[2])
    x = keras.layers.Add()([x0, x1]) + x2

    mainMod = keras.Model(inputs=inp_1, outputs=x)

    mainMod.compile(optimizer="adam", loss="mse")

    # Example input (1 sample with 32 features)
    x_train = np.random.random(size=(1, 32))  # Shape (1, 32)

    # Example target (1 sample, single output value)
    y_train = np.random.random(size=(1,))  # Shape (1,)

    # Fit the model with 1 sample
    mainMod.fit(x=x_train, y=y_train, epochs=1)

    unnestedModel = Unnest.unnestModel(mainMod)
    unnestedModel.save("./models/Unnested.keras")

    unnestedModel = keras.saving.load_model("./models/Unnested.keras")

    subModels = Split.modelSplit(unnestedModel, maxLayerNum=10)
    subMod: keras.Model
    for idx, subMod in enumerate(subModels):
        subMod.save(f"./models/SubMod_{idx}.keras")

    # print(keras.saving.load_model("./models/SubMod_1.keras").input)


def main_3():
    images = tf.ones(shape=(1, 512, 512, 3))

    model = keras.saving.load_model("./models/UnnestedYolo.keras")
    wholeModelOutput = model(images)

    subModels: list[keras.Model] = Split.modelSplit(model, maxLayerNum=50)
    for idx, subMod in enumerate(subModels):
        subMod.save(f"./models/SubYolo_{idx}.keras")

    loadedModels = []
    for idx in range(0, 9):
        loadedModels.append(keras.saving.load_model(f"./models/SubYolo_{idx}.keras"))

    producedOutputs: dict[str] = {}
    producedOutputs["input_layer_1_0_0"] = images

    for idx, subMod in enumerate(loadedModels):
        print(f"Running Model Part >>> {idx}")
        # print(subMod.input)
        subModInput: dict[str] = {}
        for inputName in subMod.input:
            subModInput[inputName] = producedOutputs[inputName]

        subModOut: dict[str] = subMod(subModInput)
        for outName in subModOut:
            producedOutputs[outName] = subModOut[outName]

    areEqual = np.array_equal(producedOutputs["box_0"], wholeModelOutput["box_0"])
    print(f"Same Outputs >>> {areEqual}")


def testSavedModel():
    kerasModel = keras.saving.load_model("./models/UnnestedYolo.keras")
    subModels: list[keras.Model] = Split.modelSplit(kerasModel, maxLayerNum=50)

    for idx, subMod in enumerate(subModels):
        subMod.save(f"./models/SubYolo_{idx}.keras")

    for idx in range(0, 9):
        print("Loaded ", idx)
        keras.saving.load_model(f"./models/SubYolo_{idx}.keras")

    # for idx in range(0, 9):
    #     loadedModel: keras.Model = keras.saving.load_model(
    #         f"./models/SubYolo_{idx}.keras"
    #     )
    #     print("Inputs >>> ", loadedModel.input)

    # print(loadedModel.signatures)


if __name__ == "__main__":
    main_3()
    # testSavedModel()
