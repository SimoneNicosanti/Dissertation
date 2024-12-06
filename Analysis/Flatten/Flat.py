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
                            ## Multiple inputs to the model
                            ## Each input layer has one input tensor --> Take them in order
                            allOpsDict[subOpName] = OperationWrapper(
                                keras.layers.Identity(name=subOpName),
                                [arg[idx]],
                                [arg[idx]],
                                [arg[idx]],
                            )
                            idx += 1
                        else:
                            ## Only one input
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
                ## Need to find the input layer corresponding as next for the current node
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
    print(f"Processing {opName} || Input >>> {opInput}")
    opOutput = operation(*opInput)
    producedOutputs[opName] = convertToList(opOutput)


def reconstructModel(allOpsDict, prevOpsDict, inputOpsList, outputOpsList):
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


def findConnections(model: keras.Model):
    allOpsDict, nextOpsDict, inputOpsList, outputOpsList = findNextConnections(model)
    prevOpsDict = findPrevConnections(nextOpsDict)

    return allOpsDict, prevOpsDict, inputOpsList, outputOpsList


def unpackModel(model: keras.Model) -> keras.Model:
    allOpsDict, prevOpsDict, inputOpsList, outputOpsList = findConnections(model)

    newModel = reconstructModel(allOpsDict, prevOpsDict, inputOpsList, outputOpsList)

    return newModel
