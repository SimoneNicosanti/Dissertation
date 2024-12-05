import pickle

import keras
import keras.src
import keras.src.ops.node
import numpy as np
import tensorflow as tf
from keras.src.ops.node import Node as KerasNode


def findAllLayersRecursive(model: keras.Model, layerPath: str, layerList: list):
    for layer in model.layers:
        # layer.name = layerPath + "/" + layer.name
        if isinstance(layer, keras.Model):
            layerList.append(layer)
            findAllLayersRecursive(layer, layerPath + "/" + layer.name, layerList)
        elif isinstance(layer, keras.Layer):
            layerList.append(layer)


def findLayerByName(allLayerList: list[keras.Layer], layerName: str):
    for layer in allLayerList:
        if layer.name == layerName:
            return layer
    return None


def findLayersConnections(allLayerList: list[keras.Layer]):
    layerNames = [layer.name for layer in allLayerList]

    prevLayersDict = {layer.name: set() for layer in allLayerList}
    nextLayersDict = {layer.name: set() for layer in allLayerList}

    for currLayer in allLayerList:
        nextOps = currLayer._outbound_nodes
        nextLayers = []
        findNextLayers(nextOps, nextLayers, layerNames)

        nextLayersDict[currLayer.name] = set(nextLayers)

        for nextLayer in nextLayers:
            prevLayersDict[nextLayer].add(currLayer.name)

    return prevLayersDict, nextLayersDict


def findNextLayers(nextOps: list[KerasNode], prevLayers: list, layerNames: list):
    for nextNode in nextOps:
        nextOperationName = nextNode.operation.name
        if nextOperationName in layerNames:
            ## The operation is a layer
            prevLayers.append(nextOperationName)
        else:
            ## The operation is not a layer --> Find the first valid layer after
            nextOpsNext = nextNode.operation._outbound_nodes
            findNextLayers(nextOpsNext, prevLayers, layerNames)


def modelParse(model: keras.Model, maxLayerNum=25) -> list[keras.Model]:
    allLayerList = model.layers
    # findAllLayersRecursive(model, "", allLayerList)

    prevOpsDict, nextOpsDict = findLayersConnections(allLayerList)
    subModels = []
    modIdx = 0
    for i in range(0, len(allLayerList), maxLayerNum):
        subLayers = allLayerList[i : min(len(allLayerList), i + maxLayerNum)]
        subLayersNames = [x.name for x in subLayers]

        subModelInput = buildSubModelInput(
            subLayers, subLayersNames, allLayerList, prevOpsDict
        )
        subModelOutput = buildSubModelOutput(subLayers, subLayersNames, nextOpsDict)
        subModel = keras.Model(inputs=subModelInput, outputs=subModelOutput)
        subModels.append(subModel)

        modIdx += 1

    return subModels


def buildSubModelInput(
    subLayers, subLayersNames, allLayerList: list[keras.Layer], prevOpsDict
):
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
                prevLayer = findLayerByName(allLayerList, prevLayerName)
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
        return inOut.values()


def buildSubModelOutput(subLayers, subLayersNames, nextOpsDict):
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
