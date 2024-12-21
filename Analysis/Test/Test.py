import keras


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


myDense = keras.layers.Dense(units=32)
subMod = subModel_2()

inp_1 = keras.Input(shape=(32,))
x = subModel(myDense)(inp_1)
x = subMod(x)
x = subMod(x)
x = myDense(x)

model = keras.Model(inputs=inp_1, outputs=x)
model.summary()
model.save("./models/Toy_1.keras")

print(model.get_layer("sequential")._inbound_nodes)
print(model.get_layer("sequential")._inbound_nodes[0].input_tensors)
print(model.get_layer("sequential")._inbound_nodes[0].operation.layers[0]._inbound_nodes[0].input_tensors)
# print(model.operations)
# print(dir(model.get_layer("functional").get_layer("dense")))
# print(model.get_layer("functional").get_layer("dense")._path)
# print(model.get_layer("dense_2").input._keras_history.operation._inbound_nodes[1].outputs)
# hist = model.output._keras_history
# operation, nodeIdx, tensIdx = hist.operation, hist.node_index, hist.tensor_index
# print(hist)
# prev = operation._inbound_nodes[nodeIdx].output_tensors[tensIdx]
# print(prev)