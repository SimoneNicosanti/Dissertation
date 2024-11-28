import pickle

import keras
import numpy as np
import tensorflow as tf


def findLayersConnections(model: keras.Model):

    layerNames = [layer.name for layer in model.layers]

    prevLayersDict = {layer.name: [] for layer in model.layers}
    nextLayersDict = {layer.name: [] for layer in model.layers}

    for currLayer in model.layers:
        currLayerInputs = (
            currLayer.input if isinstance(currLayer.input, list) else [currLayer.input]
        )
        prevLayers = []
        findPrevLayers(currLayerInputs, prevLayers, layerNames)

        prevLayersDict[currLayer.name] = prevLayers

        for prevLayer in prevLayers:
            nextLayersDict[prevLayer].append(currLayer.name)

    return prevLayersDict, nextLayersDict


def findPrevLayers(
    currLayerInputs: list[keras.KerasTensor], prevLayers: list, layerNames: list
):
    for layerInput in currLayerInputs:
        kerasHistory = getattr(layerInput, "_keras_history", None)
        if kerasHistory is not None:
            operation = kerasHistory.operation
            operationName = kerasHistory.operation.name
            if operationName in layerNames:
                ## The operation is a layer
                prevLayers.append(operationName)
            else:
                ## The operation is not a layer --> Find the first valid layer before
                opInputs = (
                    operation.input
                    if isinstance(operation.input, list)
                    else [operation.input]
                )
                findPrevLayers(opInputs, prevLayers, layerNames)


def modelParse(model: keras.Model, parts=2) -> list[keras.Model]:
    prevOpsDict, nextOpsDict = findLayersConnections(model)

    maxLayerNum = len(model.layers) // 2
    subModels = []
    for i in range(0, len(model.layers), maxLayerNum):
        subLayers = model.layers[i : min(len(model.layers), i + maxLayerNum)]
        subLayersNames = [x.name for x in subLayers]

        subModelInput = buildSubModelInput(
            subLayers, subLayersNames, model, prevOpsDict
        )
        subModelOutput = buildSubModelOutput(
            subLayers, subLayersNames, model, nextOpsDict
        )

        subModel = keras.Model(inputs=subModelInput, outputs=subModelOutput)
        subModels.append(subModel)

    return subModels


def buildSubModelInput(subLayers, subLayersNames, model, prevOpsDict):
    subModelInput = {}
    for layer in subLayers:
        if len(prevOpsDict[layer.name]) == 0:
            ## Input Layer
            newInput = keras.Input(
                shape=layer.output.shape[1:],
                tensor=layer.output,
                name=layer.name,
            )
            newInput.name = layer.name
            subModelInput[layer.name] = newInput
        else:
            for prevLayerName in prevOpsDict[layer.name]:
                prevLayerOut = model.get_layer(prevLayerName).output
                # print(prevLayerOut)
                if prevLayerName not in subLayersNames:
                    # newInput = keras.Input(tensor=prevLayerOut)
                    newInput = keras.Input(
                        shape=prevLayerOut.shape[1:],
                        tensor=prevLayerOut,
                        name=prevLayerName,
                    )
                    newInput.name = prevLayerName
                    subModelInput[prevLayerName] = newInput

    return subModelInput


def buildSubModelOutput(subLayers, subLayersNames, model, nextOpsDict):
    subModelOutput = {}
    for layer in subLayers:
        layerOutput = layer.output
        if len(nextOpsDict[layer.name]) == 0:
            ## Output Layer
            subModelOutput[layer.name] = layerOutput
        else:
            for nextLayer in nextOpsDict[layer.name]:
                if nextLayer not in subLayersNames:
                    subModelOutput[layer.name] = layerOutput
    return subModelOutput  # if len(subModelOutput) > 1 else subModelOutput[0]


def main():
    model: keras.Model = keras.applications.MobileNetV3Large()
    modelParse(model)
    testElem = readTestElem()

    x = {"input_layer": testElem}
    for i in range(0, 5):
        loadedModel = tf.saved_model.load(f"./models/SubModel_{i}")
        # loadedModel = keras.saving.load_model(f"./models/SubModel_{i}.keras")
        signature = loadedModel.signatures["serve"]
        x = signature(**x)

    predictions = x["predictions"]
    for row in predictions:
        print(f"Predicted Class >>> {np.argmax(row)}")


def readTestElem():
    testElem = None
    with open("boef_pre.pkl", "rb") as f:
        testElem = pickle.load(f)

    return tf.convert_to_tensor(value=testElem, dtype=tf.float32)


if __name__ == "__main__":
    main()
