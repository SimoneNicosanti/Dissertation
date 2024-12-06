import inspect
from typing import Callable

import keras
import keras.src
import keras.src.ops.function
import keras.src.ops.numpy
import keras_cv
import numpy as np
import tensorflow as tf
from keras.src.ops.symbolic_arguments import SymbolicArguments


class OperationWrapper:
    def __init__(self, operation, flatArguments, args, kwargs):
        self.operation = operation
        self.flatArguments = flatArguments
        self.args = args
        self.kwargs = kwargs


def findPrevConnections(nextConnectionsDict: dict[str, set[str]]):
    prevConnectionsDict: dict[str, set[str]] = {}
    for opName in nextConnectionsDict.keys():
        prevConnections = set()
        for otherOpName in nextConnectionsDict.keys():
            otherOpNexts = nextConnectionsDict[otherOpName]
            if opName != otherOpName:
                for elem in otherOpNexts:
                    if elem == opName:
                        prevConnections.add(otherOpName)
        prevConnectionsDict[opName] = prevConnections
    return prevConnectionsDict


def findNextConnections(model: keras.Model):
    nextOpsDict: dict[str, list[str]] = {}
    inputOpsList: list[str] = []
    outputOpsList: list[str] = []
    allOpsDict: dict[str, Callable] = {}

    opsQueue = model.operations

    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            ## Init queue
            opsQueue.append(layer)
            inputOpsList.append(layer.name)

    while opsQueue:
        currOp = opsQueue.pop(0)

        ## Check if it is an output op --> Add it to the list
        if currOp.name in model.output_names and currOp.name not in outputOpsList:
            outputOpsList.append(currOp.name)

        if isinstance(currOp, keras.Model):
            ## It is a sub model
            subAllOpsDict, subNextConns, subInputOps, subOutputOps = (
                findNextConnections(currOp)
            )

            ## Unify ops dicts
            idx = 0
            for subOpName in subAllOpsDict.keys():
                if subOpName not in subInputOps:
                    ## Add directly the Operation
                    allOpsDict[subOpName] = subAllOpsDict[subOpName]
                else:
                    ## It is an input layer --> Change it with Identity
                    for arg in currOp._inbound_nodes[0].arguments.args:
                        if isinstance(arg, list):
                            allOpsDict[subOpName] = OperationWrapper(
                                keras.layers.Identity(name=subOpName),
                                [arg[idx]],
                                [arg[idx]],
                                [arg[idx]],
                            )
                            idx += 1
                        else:
                            allOpsDict[subOpName] = OperationWrapper(
                                keras.layers.Identity(name=subOpName),
                                [arg],
                                [arg],
                                [arg],
                            )

            ## Unify next conns dict
            for subOpName in subNextConns.keys():
                nextOpsDict.setdefault(subOpName, [])
                nextOpsDict[subOpName].extend(subNextConns[subOpName])

            ## Adding sub model place holder
            allOpsDict[currOp.name] = OperationWrapper(
                keras.layers.Identity(name=currOp.name),
                [currOp.outputs],
                [currOp.outputs],
                None,
            )

            ## Connecting sub model output to sub model Identity placeholder
            for subOutOpName in subOutputOps:
                nextOpsDict.setdefault(subOutOpName, [])
                nextOpsDict[subOutOpName].extend([currOp.name])

        else:
            ## It is a normal layer
            allOpsDict[currOp.name] = OperationWrapper(
                currOp,
                currOp._inbound_nodes[0].arguments._flat_arguments,
                currOp._inbound_nodes[0].arguments.args,
                currOp._inbound_nodes[0].arguments.kwargs,
            )

        ## Common management!! If it is a sub model we added a placeholder with the same name
        # nextOpsDict[currOp.name] = set()

        currOpNextOps = [node.operation for node in currOp._outbound_nodes]
        nextOpsDict.setdefault(currOp.name, [])
        for nextOp in currOpNextOps:
            if not isinstance(nextOp, keras.Model):
                nextOpsDict[currOp.name].append(nextOp.name)
            else:
                nextOpsDict[currOp.name].extend(
                    findSubModelCorrespondingInputLayer(nextOp, currOp.name)
                )

            ## If is a keras model I will find its connections when parsing sub model

    convertedNextOpsDicts: dict[str, set[str]] = {}
    for key in nextOpsDict:
        convertedNextOpsDicts[key] = set(nextOpsDict[key])

    return allOpsDict, convertedNextOpsDicts, inputOpsList, outputOpsList


def findSubModelCorrespondingInputLayer(
    subModel: keras.Model, currOpName: str
) -> list[str]:
    layerInputs = subModel._inbound_nodes[0].arguments._flat_arguments
    layerInputsPrevNames = [
        inp._keras_history.operation.name
        for inp in layerInputs
        if isinstance(inp, keras.KerasTensor)
    ]
    subModInputs = subModel.inputs
    subModelInputLayers = [inp._keras_history.operation.name for inp in subModInputs]

    correspondingLayers = []
    for idx, elem in enumerate(layerInputsPrevNames):
        if elem == currOpName:
            correspondingLayers.append(subModelInputLayers[idx])
    return correspondingLayers


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


def unpackArguments(args, producedOutputs):
    opInput = []
    for arg in args:
        if arg is None:
            continue

        if isinstance(arg, keras.KerasTensor):
            hist = arg._keras_history
            prevOpName, tensorIndex = hist.operation.name, hist.tensor_index
            prevOpOutputs = producedOutputs[prevOpName]
            opInput.append(prevOpOutputs[tensorIndex])
        elif isinstance(arg, list):
            opInput.append(unpackArguments(arg, producedOutputs))
        else:
            opInput.append(arg)
    return opInput


def convertToList(opOutput):
    if isinstance(opOutput, list):
        return opOutput
    elif isinstance(opOutput, dict):
        return opOutput.values()
    else:
        return [opOutput]


def runOperation(opName: str, allOpsDict, prevOpsDict, producedOutputs):
    for prevOpName in prevOpsDict[opName]:
        if prevOpName not in producedOutputs:
            runOperation(prevOpName, allOpsDict, prevOpsDict, producedOutputs)
    operationWrapper: OperationWrapper = allOpsDict[opName]
    operation = operationWrapper.operation
    opInput = unpackArguments(operationWrapper.args, producedOutputs)
    if opName == "get_item_1":
        print("ITEM ARGS >>> ", type(operationWrapper.args[1][0]))
    print(opName, opInput)
    opOutput = operation(*opInput)
    producedOutputs[opName] = convertToList(opOutput)


def unpackModel(allOpsDict, prevOpsDict, nextOpsDict, inputOpsList, outputOpsList):
    producedOutputs = {}
    for inpLayerName in inputOpsList:
        outputList = convertToList(allOpsDict[inpLayerName].operation.output)
        producedOutputs[inpLayerName] = outputList
        # print("Produced >>> ", producedOutputs[inpLayerName])

    for opName in allOpsDict:
        if opName not in producedOutputs:
            runOperation(opName, allOpsDict, prevOpsDict, producedOutputs)

    newModelInput = []
    for inp in [producedOutputs[inputName] for inputName in inputOpsList]:
        newModelInput.extend(inp)
    newModelOutput = []
    for out in [producedOutputs[outputName] for outputName in outputOpsList]:
        newModelOutput.extend(out)
    newModel = keras.Model(inputs=newModelInput, outputs=newModelOutput)

    return newModel


def main():

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

    mainMod.save("./models/Model.keras")

    print(
        mainMod.get_layer("functional_1")
        .get_layer("functional")
        ._inbound_nodes[0]
        .arguments.args
    )

    allOpsDict, nextOpsDict, inputOpsList, outputOpsList = findNextConnections(mainMod)
    prevOpsDict = findPrevConnections(nextOpsDict)

    print(nextOpsDict)
    print()
    print(prevOpsDict)

    unpackedModel = unpackModel(
        allOpsDict, prevOpsDict, nextOpsDict, inputOpsList, outputOpsList
    )
    unpackedModel.save("./models/Unpacked.keras")

    # print(unpackedModel.outputs)

    pred_1 = mainMod.predict(x_train)
    pred_2 = unpackedModel.predict(x_train)
    print(pred_1, pred_2)
    print(np.array_equal(pred_1, pred_2))


def main_1():
    keras.src.ops.numpy.GetItem()
    images = tf.ones(shape=(1, 512, 512, 3))
    labels = {
        "boxes": tf.constant(
            [
                [
                    [0, 0, 100, 100],
                    [100, 100, 200, 200],
                    [300, 300, 100, 100],
                ]
            ],
            dtype=tf.float32,
        ),
        "classes": tf.constant([[1, 1, 1]], dtype=tf.int64),
    }

    model = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_m_backbone_coco")

    model = keras_cv.models.YOLOV8Detector(
        num_classes=20,
        bounding_box_format="xywh",
        backbone=model,
        fpn_depth=2,
    )

    # Evaluate model without box decoding and NMS
    model(images)

    # Train model
    # model.compile(
    #     classification_loss="binary_crossentropy",
    #     box_loss="ciou",
    #     optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
    #     jit_compile=False,
    # )
    # model.fit(images, labels)

    allOpsDict, nextOpsDict, inputOpsList, outputOpsList = findNextConnections(model)
    prevOpsDict = findPrevConnections(nextOpsDict)

    print(model.output)

    unpackedModel = unpackModel(
        allOpsDict, prevOpsDict, nextOpsDict, inputOpsList, outputOpsList
    )
    unpackedModel.save("./models/Unpacked.keras")

    # for op in model.operations:
    #     if op.name == "get_item":
    #         print(op.output)
    #         break

    # for op in unpackedModel.operations:
    #     if op.name == "get_item":
    #         print(op.output)
    #         break

    # loaded = keras.models.load_model("./models/Unpacked.keras")

    # print(model.output_names)
    # print(unpackedModel.output_names)

    pred_1 = model.predict(images)
    pred_2 = unpackedModel.predict(images)
    pred_2 = {"boxes": pred_2[0], "classes": pred_2[1]}

    decoded = model.decode_predictions(pred_2, images)

    # print(pred_1["boxes"])
    # print(pred_2[0])
    print(np.array_equal(pred_1["classes"], decoded["classes"]))


if __name__ == "__main__":
    main_1()
