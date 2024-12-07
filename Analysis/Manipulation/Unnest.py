from typing import Callable

import keras
import keras.src
import keras.src.ops.function
import keras.src.ops.numpy
import Utils


def findSubModels(model: keras.Model) -> list[keras.Model]:
    subModels: list[keras.Model] = []
    opsQueue: list[keras.Operation] = model.operations

    while opsQueue:
        currOp: keras.Operation = opsQueue.pop()
        if isinstance(currOp, keras.Model):
            subSubModels: list[keras.Model] = findSubModels(currOp)
            subModels.extend(subSubModels)

            subModels.append(currOp)
    return subModels


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


def convertToList(anyValue):
    if isinstance(anyValue, list):
        return anyValue
    elif isinstance(anyValue, dict):
        return anyValue.values()
    else:
        return [anyValue]


def wrapOperation(operation: keras.Operation) -> keras.Operation:

    ## In order to keep the model original struct, we change both
    ## input layers and layers representing sub models with IdentityLayers
    newOperation: keras.Operation = None
    if isinstance(operation, keras.Model) or isinstance(
        operation, keras.layers.InputLayer
    ):
        newOperation = keras.layers.Identity(name=operation.name)
    else:
        newOperation = operation
    return newOperation


def findArguments(operation: keras.Operation, allSubModels: list[keras.Model]) -> list:

    if isinstance(operation, keras.Model):
        ## It is a sub model
        ## We change the sub model with an Identity Layer
        ## returning the same output as the sub model itself
        return [operation.outputs]
    elif isinstance(operation, keras.layers.InputLayer):
        ## It is input layer of sub model
        ## We chnage it with an Identity layer returning
        ## the same output as the sub model
        opSubModel: keras.Model = None
        inputIdx: int = None
        for subMod in allSubModels:
            subModInputs: list[str] = Utils.findInputLayers(subMod)
            if operation.name in subModInputs:
                inputIdx = subModInputs.index(operation.name)
                opSubModel = subMod
                break

        ## TODO >> Check this if is enough general
        for argElem in opSubModel._inbound_nodes[0].arguments.args:
            if isinstance(argElem, list):
                return [argElem[inputIdx]]
            else:
                return [argElem]

    else:
        ## Simple operation
        ## Return its args
        return operation._inbound_nodes[0].arguments.args


def runOperation(
    opName: str,
    allOpsDict: dict[str, keras.Operation],
    prevOpsDict: dict[str, set[str]],
    producedOutputs: dict[str, list[keras.KerasTensor]],
    allSubModels: list[keras.Model],
):
    ## Run all needed previous operations
    for prevOpName in prevOpsDict[opName]:
        if prevOpName not in producedOutputs:
            runOperation(
                prevOpName, allOpsDict, prevOpsDict, producedOutputs, allSubModels
            )

    toCall: keras.Operation = wrapOperation(allOpsDict[opName])
    callArgs: list = findArguments(allOpsDict[opName], allSubModels)

    opInput = unpackArguments(callArgs, producedOutputs)
    print(f"Processing {opName} || Input >>> {opInput}")
    opOutput = toCall(*opInput)
    producedOutputs[opName] = convertToList(opOutput)


def reconstructModel(
    allOpsDict: dict[str, keras.Operation],
    prevOpsDict: dict[str, set[str]],
    inputOpsList: list[str],
    outputOpsList: list[str],
    allSubModels: list[keras.Model],
):
    producedOutputs: dict[str, list[keras.KerasTensor]] = {}
    for inpLayerName in inputOpsList:
        outputList = convertToList(allOpsDict[inpLayerName].output)
        producedOutputs[inpLayerName] = outputList
        # print("Produced >>> ", producedOutputs[inpLayerName])

    for opName in allOpsDict:
        if opName not in producedOutputs:
            runOperation(opName, allOpsDict, prevOpsDict, producedOutputs, allSubModels)

    newModelInput = []
    for inp in [producedOutputs[inputName] for inputName in inputOpsList]:
        newModelInput.extend(inp)
    newModelOutput = []
    for out in [producedOutputs[outputName] for outputName in outputOpsList]:
        newModelOutput.extend(out)
    newModel = keras.Model(inputs=newModelInput, outputs=newModelOutput)

    return newModel


def unnestModel(model: keras.Model) -> keras.Model:
    prevOpsDict: dict[str, set[str]] = Utils.findPrevConnections(model)
    allOpsDict: dict[str, keras.Operation] = Utils.findAllOps(model)
    inputOpsList: list[str] = Utils.findInputLayers(model)
    outputOpsList: list[str] = model.output_names
    allSubModels: list[keras.Model] = findSubModels(model)

    newModel = reconstructModel(
        allOpsDict, prevOpsDict, inputOpsList, outputOpsList, allSubModels
    )

    for op in newModel.operations:
        op.name = f"unpacked_{op.name}"

    return newModel
