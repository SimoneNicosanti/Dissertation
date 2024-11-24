import pickle

import keras
import tensorflow as tf


def findLayersConnections(model: keras.Model):
    config = model.get_config()
    opsList = config["layers"]
    opsConfigDict = {}
    for opConfig in opsList:
        opName = opConfig["name"]
        opsConfigDict[opName] = opConfig

    layerNames = [layer.name for layer in model.layers]

    prevOpsDict = {}
    nextOpsDict = {}
    for layerName in layerNames:
        prevOpsDict[layerName] = []
        nextOpsDict[layerName] = []

    for layerName in layerNames:
        currConfig = opsConfigDict[layerName]
        prevLayers = findPrevLayersFromConfig(currConfig, opsConfigDict, layerNames)

        prevOpsDict[layerName] = prevLayers

        for prevLayer in prevLayers:
            nextOpsDict[prevLayer].append(layerName)

    return prevOpsDict, nextOpsDict


def modelParse(model: keras.Model):
    prevOpsDict, nextOpsDict = findLayersConnections(model)

    maxLayerNum = 30
    modelIdx = 0
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
        subModel.save(f"./models/SubModel_{modelIdx}.keras")
        with open(f"./models/SubModel_{modelIdx}.json", "w") as f:
            f.write(str(subModel.get_config()))

        modelIdx += 1


def findPrevLayersFromConfig(currConfig, configDict, layerNames):
    prevOpsNames = []
    prevLayers = []
    findHistoryInDepth(currConfig["inbound_nodes"], prevOpsNames)
    for opName in prevOpsNames:
        if opName in layerNames:
            ## It is a valid layer --> Append to prev Layers
            prevLayers.append(opName)
        else:
            ## It is other operation --> Find first valid prev layer
            opConfig = configDict[opName]
            opPrevLayers = findPrevLayersFromConfig(opConfig, configDict, layerNames)
            prevLayers += opPrevLayers

    return prevLayers


def findHistoryInDepth(node, histories: list):
    if isinstance(node, dict):
        ## Check if key is present
        if "keras_history" in node:
            histories.append(node["keras_history"][0])  ## Takes operation name
        else:
            for key in node:
                findHistoryInDepth(node[key], histories)
    elif isinstance(node, list) or isinstance(node, tuple):
        ## Check each elem for the key
        for elem in node:
            findHistoryInDepth(elem, histories)

    return


def buildSubModelInput(subLayers, subLayersNames, model, prevOpsDict):
    subModelInput = []
    for layer in subLayers:
        if len(prevOpsDict[layer.name]) == 0:
            ## Input Layer
            subModelInput.append(layer.output)
        else:
            for prevLayer in prevOpsDict[layer.name]:
                if prevLayer not in subLayersNames:
                    prevLayerOut = model.get_layer(prevLayer).output
                    if isinstance(prevLayerOut, list):
                        subModelInput += [
                            keras.Input(
                                shape=prevLayerOut[i].shape[1:],
                                tensor=prevLayerOut[i],
                                name=prevLayer,
                            )
                            for i in range(0, len(prevLayerOut))
                        ]
                    else:
                        subModelInput.append(
                            keras.Input(
                                shape=prevLayerOut.shape[1:],
                                tensor=prevLayerOut,
                                name=prevLayer,
                            )
                        )
    return subModelInput if len(subModelInput) > 1 else subModelInput[0]


def buildSubModelOutput(subLayers, subLayersNames, model, nextOpsDict):
    subModelOutput = []
    for layer in subLayers:
        if len(nextOpsDict[layer.name]) == 0:
            ## Output Layer
            subModelOutput.append(layer.output)
        else:
            for nextLayer in nextOpsDict[layer.name]:
                if nextLayer not in subLayersNames:
                    nextLayerIn = model.get_layer(layer.name).output
                    if isinstance(nextLayerIn, list):
                        subModelOutput += nextLayerIn
                    else:
                        subModelOutput.append(nextLayerIn)
    return subModelOutput if len(subModelOutput) > 1 else subModelOutput[0]


def main():
    model: keras.Model = keras.applications.MobileNetV3Large()
    modelParse(model)
    loadedModel_0: keras.Model = keras.saving.load_model("./models/SubModel_0.keras")
    loadedModel_1: keras.Model = keras.saving.load_model("./models/SubModel_1.keras")
    loadedModel_2: keras.Model = keras.saving.load_model("./models/SubModel_2.keras")

    testElem = readTestElem()


def readTestElem():
    testElem = None
    with open("boef_pre.pkl", "rb") as f:
        testElem = pickle.load(f)

    return tf.convert_to_tensor(value=testElem)


if __name__ == "__main__":
    main()
