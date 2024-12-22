import unittest

import keras
import tensorflow as tf
from Manipulation import Unnester, Utils
import numpy as np

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
    x2 = keras.layers.Dense(units=32)(x1)
    mod_1 = keras.Model(inputs=inp, outputs=x2)
    return mod_1


class RepeatTest(unittest.TestCase):

    def test_toy(self):
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

        toy.save("./models/RepeatedToy.keras")

        test_elem = tf.ones(shape = (1, 32))
        out_1 = toy(test_elem)
        
        unnestedModel = Unnester.unnestModel(toy)
        unnestedModel.save("./models/UnnestedRepeatedToy.keras")

        loadedModel = keras.saving.load_model("./models/UnnestedRepeatedToy.keras")
        out_2 = loadedModel(test_elem)["dense_0"]

        normDiff = tf.norm(out_1 - out_2)

        self.assertAlmostEqual(normDiff, 0, delta = 1.e-3)


if __name__ == "__main__" :
    unittest.main()