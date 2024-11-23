import keras


def sumFn(x):
    return x + 3


model: keras.Model = keras.applications.MobileNetV3Large()

subModel = keras.Model(
    inputs=[
        model.get_layer("expanded_conv_11_squeeze_excite_conv").input,
        model.get_layer("activation_12").output,
    ],
    outputs=model.get_layer("expanded_conv_11_squeeze_excite_mul").output,
)

subModel.save("./attempt.keras")
