import keras
import numpy as np
import tensorflow as tf
from Manipulation import Unnester

from Flops.FlopsComputer import FlopsComputer


def subModel_1():
    inp_1 = keras.layers.Input(shape=(32,))
    inp_2 = keras.layers.Input(shape=(32,))
    x1 = keras.layers.Dense(units=32)(inp_1)
    x2 = keras.layers.Dense(units=32)(inp_2)
    x3 = x1 + x2
    return keras.Model(inputs=[inp_1, inp_2], outputs=x3)


def subModel_2():
    return keras.Sequential(
        layers=[
            keras.layers.Input(shape=(32,)),
            keras.layers.Dense(units=32),
            keras.layers.Dense(units=32),
            keras.layers.Dense(units=32),
        ]
    )


def subModel(myDense):
    inp = keras.layers.Input(shape=(32,))
    x1 = myDense(inp)
    x1 = myDense(x1)
    x2 = keras.layers.Dense(units=32)(x1)
    mod_1 = keras.Model(inputs=inp, outputs=x2)
    return mod_1


def build_toy():
    myDense = keras.layers.Dense(units=32)
    subMod = subModel_2()

    inp_1 = keras.Input(shape=(32,))
    x = subModel(myDense)(inp_1)
    x = subMod(x)
    x = subMod(x)
    x = myDense(x)

    toy = keras.Model(inputs=inp_1, outputs=x)
    toy.compile(optimizer="adam", loss="mse")

    x_train = np.random.random(size=(1, 32))  # Shape (1, 32)
    y_train = np.random.random(size=(1,))  # Shape (1,)
    toy.fit(x=x_train, y=y_train, epochs=1)

    return toy


def test_toy():
    toy = build_toy()

    test_elem = tf.ones(shape=(1, 32))
    out_1 = toy(test_elem)

    unnestedModel = Unnester.unnestModel(toy)
    unnestedModel.save("./models/UnnestedRepeatedToy.keras")

    loadedModel = keras.saving.load_model("./models/UnnestedRepeatedToy.keras")

    out_2 = loadedModel(test_elem)["dense_0"]

    normDiff = tf.norm(out_1 - out_2)
    print(f"Norm of Diff {normDiff}")


def test_flops():
    toy = build_toy()
    toy.save("./models/RepeatedToy.keras")
    print(toy.get_layer("dense")._inbound_nodes[0].parent_nodes[0].parent_nodes)
    # print(dir(toy))
    # print(graph.nodePool.getAllKeys())
    print(toy.inputs)
    inpShapes = []
    ## It is always a list!!
    for _ in enumerate(toy.inputs):
        inpShapes.append((1, 32))

    comp = FlopsComputer(toy)
    comp.computeRunningTimes(inpShapes, 10)


if __name__ == "__main__":
    # test_toy()
    test_flops()
