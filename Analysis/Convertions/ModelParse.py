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


def modelParse(model: keras.Model, maxLayerNum=1) -> list[keras.Model]:
    prevOpsDict, nextOpsDict = findLayersConnections(model)
    subModels = []
    modIdx = 0
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

        modIdx += 1

    return subModels


def buildSubModelInput(subLayers, subLayersNames, model, prevOpsDict):
    subModelInput = {}
    for layer in subLayers:
        if len(prevOpsDict[layer.name]) == 0:
            ## Input Layer
            # layer.output.name = "input"
            layerOutputs = convertToList(layer.output)
            for i, inp in enumerate(layerOutputs):
                subModelInput[f"input_{i}"] = inp
        else:
            for prevLayerName in prevOpsDict[layer.name]:
                prevLayer = model.get_layer(prevLayerName)
                prevLayerOut = convertToList(prevLayer.output)
                # print(prevLayerOut)
                if prevLayerName not in subLayersNames:
                    for _, out in enumerate(prevLayerOut):
                        # newInput = keras.Input(tensor=prevLayerOut)
                        subModelInput[out.name] = out
    return subModelInput


def convertToList(inOut) -> list[keras.KerasTensor]:
    if isinstance(inOut, keras.KerasTensor):
        return [inOut]
    if isinstance(inOut, list):
        return inOut
    if isinstance(inOut, dict):
        return [inOut[key] for key in inOut]


def buildSubModelOutput(subLayers, subLayersNames, model, nextOpsDict):
    subModelOutput = {}
    for layer in subLayers:
        layerOutput = convertToList(layer.output)
        # print(layerOutput)
        if len(nextOpsDict[layer.name]) == 0:
            ## Output Layer
            for i, out in enumerate(layerOutput):
                subModelOutput[f"output_{i}"] = out
        else:
            for nextLayer in nextOpsDict[layer.name]:
                if nextLayer not in subLayersNames:
                    for _, out in enumerate(layerOutput):
                        subModelOutput[out.name] = out
    return subModelOutput


def main():
    model: keras.Model = keras.applications.MobileNetV3Large()
    subModels = modelParse(model, maxLayerNum=40)

    modIdx = 0
    for subMod in subModels:
        # print(subMod.input)
        # print(subMod.output)
        # print()
        subMod.save(f"./models/SubModel_{modIdx}.keras")
        modIdx += 1

    x = {"input": readTestElem()}
    for i in range(0, 5):
        # loadedModel = tf.saved_model.load(f"./models/SubModel_{i}")
        loadedModel = keras.saving.load_model(f"./models/SubModel_{i}.keras")
        x = loadedModel(x)

    predictions = x["output_0"]
    for row in predictions:
        print(f"Predicted Class >>> {np.argmax(row)}")


def readTestElem():
    testElem = None
    with open("boef_pre.pkl", "rb") as f:
        testElem = pickle.load(f)

    return tf.convert_to_tensor(value=testElem, dtype=tf.float32)


if __name__ == "__main__":
    import warnings

    # Suppress specific warnings
    # warnings.filterwarnings("ignore", message=".*tensor name.*")

    main()
