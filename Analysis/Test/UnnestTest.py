import keras
import keras_cv
import numpy as np
import tensorflow as tf
from Manipulation import Unnest, Unnester
from Manipulation.ModelGraph import ModelGraph


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


def subModel():
    inp = keras.layers.Input(shape=(32,))
    x1 = keras.layers.Dense(units=32)(inp)
    x2 = keras.layers.Dense(units=32)(inp)
    x3 = subModel_1()([x1, x2])
    x4 = subModel_2()(x3)
    x5 = keras.layers.Dense(units=32)(x4)
    mod_1 = keras.Model(inputs=inp, outputs=[x1, x2, x5])
    return mod_1


def main():

    inp_1 = keras.Input(shape=(32,))
    x = subModel()(inp_1)
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

    # print(Utils.findNextConnections(mainMod))

    unnestedModel = Unnester.unnestModel(mainMod)
    unnestedModel.save("./models/UnnestedToy.keras")

    # pred_1 = mainMod.predict(x_train)
    # pred_2 = unnestedModel.predict(x_train)
    # print(pred_1, pred_2)
    # print(np.array_equal(pred_1, pred_2["add_2_0"]))


def main_1():
    images = tf.ones(shape=(5, 512, 512, 3))

    model = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_l_backbone_coco")

    model = keras_cv.models.YOLOV8Detector(
        num_classes=20,
        bounding_box_format="xywh",
        backbone=model,
        fpn_depth=2,
    )

    # Evaluate model without box decoding and NMS
    model(images)
    pred_1 = model(images)

    unnestedModel: keras.Model = Unnester.unnestModel(model)
    print(unnestedModel.output)
    unnestedModel.save("./models/UnnestedYolo.keras")

    ## The differences in the outputs of predict
    ##is because predict in YoloPredictor decodifies automatically

    pred_2 = unnestedModel(images)

    print("Same Outputs >>> ", np.array_equal(pred_1["boxes"], pred_2["box_0"]))


if __name__ == "__main__":
    main_1()