import keras
from Manipulation import Utils


## Params
## @param allOpsNames : list of operation names making up the model
## @param allOpsDict : dict mapping operation name to operation object
## @param prevOpsDic : dict of prev connections
## @param nextOpsDict : dict of next connections
## @param inputOpsDict :
##          key -> input operation name ; value -> list of tensor index of operation output given as input to this model
## @param outputOpsDict :
##          key -> output operation name ; value -> list of tensor index of operation output given as output to this model
def reconstructModel(
    allOpsNames: list[str],
    allOpsDict: dict[str, keras.Operation],
    prevOpsDict: dict[str, set[str]],
    inputOpsDict: dict[str, set[int]],
    outputOpsDict: dict[str, set[int]],
    allSubModels: list[keras.Model],
) -> keras.Model:
    producedOutputs: dict[str, list[keras.KerasTensor]] = {}
    for inpLayerName in inputOpsDict.keys():
        producedOutputs.setdefault(inpLayerName, [])
        outputList = Utils.convertToList(allOpsDict[inpLayerName].output)
        for idx, out in enumerate(outputList):
            tempInpLayer = keras.layers.InputLayer(
                shape=out.shape[1:], dtype=out.dtype, name=f"{inpLayerName}_{idx}"
            )
            producedOutputs[inpLayerName].append(tempInpLayer.output)

    for opName in allOpsNames:
        if opName not in producedOutputs:
            runOperation(opName, allOpsDict, prevOpsDict, producedOutputs, allSubModels)

    newModelInput = {}
    for inpLayerName in inputOpsDict.keys():
        tensorIdxs = inputOpsDict[inpLayerName]
        for idx in tensorIdxs:
            newModelInput[f"{inpLayerName}_{idx}"] = producedOutputs[inpLayerName][idx]

    ## TODO Check if this handling of multiple outputs of layer is enough
    newModelOutput = {}
    for outLayerName in outputOpsDict.keys():
        tensorIdxs = outputOpsDict[outLayerName]
        for idx in tensorIdxs:
            newModelOutput[f"{outLayerName}_{idx}"] = producedOutputs[outLayerName][idx]

    newModel = keras.Model(inputs=newModelInput, outputs=newModelOutput)

    return newModel


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
    # print(f"Processing {opName} || Input >>> {opInput}")
    opOutput = toCall(*opInput)
    producedOutputs[opName] = Utils.convertToList(opOutput)


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
