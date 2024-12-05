from typing import Callable

import keras
import keras.src
import keras.src.ops.function
import numpy as np


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
    nextOpsDict: dict[str, set[str]] = {}
    inputOpsList: list[str] = []
    outputOpsList: list[str] = []
    allOpsDict: dict[str, Callable] = {}

    opsQueue = []

    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            ## Init queue
            opsQueue.append(layer)
            inputOpsList.append(layer.name)

    while opsQueue:
        currOp = opsQueue.pop(0)

        ## Check if it is an output op --> Add it to the list
        if currOp.name in model.output_names:
            outputOpsList.append(currOp.name)

        if isinstance(currOp, keras.Model):
            ## It is a sub model
            subAllOpsDict, subNextConns, subInputOps, subOutputOps = (
                findNextConnections(currOp)
            )

            ## Unify ops dicts
            for subOpName in subAllOpsDict.keys():
                if subOpName not in subInputOps:
                    ## Add directly the Operation
                    allOpsDict[subOpName] = subAllOpsDict[subOpName]
                else:
                    ## It is an input layer --> Change it with Identity
                    allOpsDict[subOpName] = keras.layers.Identity(name=subOpName)

            ## Unify next conns dict
            for subOpName in subNextConns.keys():
                nextOpsDict[subOpName] = subNextConns[subOpName]

            ## Connecting sub model output to next nodes in main model
            for subOutOpName in subOutputOps:
                ## Output ops of sub model are connected to next ops of current ops
                nextOpsDict[subOutOpName] = set(
                    [node.operation.name for node in currOp._outbound_nodes]
                )

        else:
            ## It is a normal layer --> Find its connections
            allOpsDict[currOp.name] = currOp
            nextOpsDict[currOp.name] = set()

            currOpNextOps = [node.operation for node in currOp._outbound_nodes]

            for nextOp in currOpNextOps:
                if not isinstance(nextOp, keras.Model):
                    nextOpsDict[currOp.name].add(nextOp.name)
                else:
                    for layer in nextOp.layers:
                        if isinstance(layer, keras.layers.InputLayer):
                            nextOpsDict[currOp.name].add(layer.name)
                ## If is a keras model I will find its connections when parsing sub model

        ## Add next ops to process queue
        opsQueue += [node.operation for node in currOp._outbound_nodes]

    return allOpsDict, nextOpsDict, inputOpsList, outputOpsList


def subModel():
    inpLayer = keras.layers.InputLayer(shape=(32,))
    x = keras.layers.Dense(units=32)(inpLayer.output)
    x1 = keras.layers.Dense(units=32)(x)
    x2 = keras.layers.Dense(units=32)(x)
    x3 = x1 + x2
    mod_1 = keras.Model(inputs=inpLayer.output, outputs=x3)
    return mod_1


def runOperation(opName: str, allOpsDict, prevOpsDict, producedOutputs):
    for prevOpName in prevOpsDict[opName]:
        if prevOpName not in producedOutputs:
            runOperation(prevOpName, allOpsDict, prevOpsDict, producedOutputs)

    opInput = [producedOutputs[prevName] for prevName in prevOpsDict[opName]]
    callable = allOpsDict[opName]
    opOutput = callable(*opInput)
    producedOutputs[opName] = opOutput


def unpackModel(allOpsDict, prevOpsDict, nextOpsDict, inputOpsList, outputOpsList):
    producedOutputs = {}
    for inpLayerName in inputOpsList:
        producedOutputs[inpLayerName] = allOpsDict[inpLayerName].output

    for opName in allOpsDict:
        if opName not in inputOpsList:
            runOperation(opName, allOpsDict, prevOpsDict, producedOutputs)

    newModelInput = [producedOutputs[inputName] for inputName in inputOpsList]
    newModelOutput = [producedOutputs[outputName] for outputName in outputOpsList]
    newModel = keras.Model(inputs=newModelInput, outputs=newModelOutput)

    return newModel


def main():

    inp_1 = keras.Input(shape=(32,))
    x = subModel()(inp_1)
    x = keras.layers.Dense(units=1)(x)
    mainMod = keras.Model(inputs=inp_1, outputs=x)

    mainMod.compile(optimizer="adam", loss="mse")

    # Example input (1 sample with 32 features)
    x_train = np.random.random(size=(1, 32))  # Shape (1, 32)

    # Example target (1 sample, single output value)
    y_train = np.random.random(size=(1,))  # Shape (1,)

    # Fit the model with 1 sample
    mainMod.fit(x=x_train, y=y_train, epochs=1)

    mainMod.save("Model.keras")

    # for layer in mainMod.layers[1].layers:
    #     print(layer.name, layer.input, layer.output)

    allOpsDict, nextOpsDict, inputOpsList, outputOpsList = findNextConnections(mainMod)
    prevOpsDict = findPrevConnections(nextOpsDict)
    print(prevOpsDict)

    unpackedModel = unpackModel(
        allOpsDict, prevOpsDict, nextOpsDict, inputOpsList, outputOpsList
    )
    unpackedModel.save("./Unpacked.keras")


if __name__ == "__main__":
    main()
