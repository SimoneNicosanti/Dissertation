import unittest

import keras
import keras_cv
import numpy as np
import tensorflow as tf
from Manipulation import Unnester


class UnnestTestCase(unittest.TestCase) :

    @unittest.skip("Have to check this")
    def test_toy_model(self) :
        toy : keras.Model = self.sub_toy()
        unnestedToy = Unnester.unnestModel(toy)
        unnestedToy.save("./models/UnnestedToy.keras")

        x_test = np.random.random(size=(1, 32))
        pred_1 = toy(x_test, training = False)
        pred_2 = unnestedToy(x_test, training = False)

        diffNorm = tf.norm(pred_1 - pred_2["add_2_0"])
        print("Toy Model Test")
        print(f"Norm of Difference >> {diffNorm}")
        self.assertAlmostEqual(diffNorm, 0, delta=1.e-3)
        

    def sub_toy_1(self):
        inp_1 = keras.layers.Input(shape=(32,))
        inp_2 = keras.layers.Input(shape=(32,))
        x1 = keras.layers.Dense(units=32)(inp_1)
        x2 = keras.layers.Dense(units=32)(inp_2)
        x3 = x1 + x2
        return keras.Model(inputs=[inp_1, inp_2], outputs=x3)


    def sub_toy_2(self):
        return keras.Sequential(
            layers=[
                keras.layers.Input(shape=(32,)),
                keras.layers.Dense(units=32),
                keras.layers.Dense(units=32),
                keras.layers.Dense(units=32),
            ]
        )


    def sub_toy(self):
        inp = keras.layers.Input(shape=(32,))
        x1 = keras.layers.Dense(units=32)(inp)
        x2 = keras.layers.Dense(units=32)(inp)
        x3 = self.sub_toy_1()([x1, x2])
        x4 = self.sub_toy_2()(x3)
        x5 = keras.layers.Dense(units=32)(x4)
        mod_1 = keras.Model(inputs=inp, outputs=[x1, x2, x5])
        return mod_1

    def toy_model(self) :
        inp_1 = keras.Input(shape=(32,))
        x = self.sub_toy()(inp_1)
        x0 = keras.layers.Dense(units=1)(x[0])
        x1 = keras.layers.Dense(units=1)(x[1])
        x2 = keras.layers.Dense(units=1)(x[2])
        x = keras.layers.Add()([x0, x1]) + x2

        mainMod = keras.Model(inputs=inp_1, outputs=x)
        mainMod.save("./models/Toy.keras")
        mainMod.compile(optimizer="adam", loss="mse")

        # Example input (1 sample with 32 features)
        x_train = np.random.random(size=(1, 32))  # Shape (1, 32)

        # Example target (1 sample, single output value)
        y_train = np.random.random(size=(1,))  # Shape (1,)

        # Fit the model with 1 sample
        mainMod.fit(x=x_train, y=y_train, epochs=1)

        return mainMod


    def test_yolo(self) :
        images = tf.ones(shape=(5, 512, 512, 3))

        backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_l_backbone_coco")

        yolo = keras_cv.models.YOLOV8Detector(
            num_classes=20,
            bounding_box_format="xywh",
            backbone=backbone,
            fpn_depth=2,
        )

        # Evaluate model without box decoding and NMS
        pred_1 = yolo(images, training = False)

        unnestedModel: keras.Model = Unnester.unnestModel(yolo)
        unnestedModel.save("./models/UnnestedYolo.keras")

        ## The differences in the outputs of predict
        ## is because predict in YoloPredictor decodifies automatically
        loadedModel : keras.Model = keras.saving.load_model("./models/UnnestedYolo.keras")
        pred_2 = loadedModel(images, training = False)

        diffNorm = tf.norm(pred_1["boxes"] - pred_2["box_0"])
        print()
        print("Yolo Model Test")
        print(f"Norm of Difference >> {diffNorm}")
        print()
        self.assertAlmostEqual(diffNorm, 0, delta=1.e-3)

    


if __name__ == '__main__':
    unittest.main()