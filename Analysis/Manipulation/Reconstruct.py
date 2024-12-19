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
    args, kwargs = findArguments(allOpsDict[opName], allSubModels)

    call_args = unpack_args(args, producedOutputs)
    call_kwargs = unpack_kwargs(kwargs, producedOutputs)
    logReconstruct(toCall, call_args, call_kwargs)

    opOutput = toCall(*call_args, **call_kwargs)

    producedOutputs[opName] = Utils.convertToList(opOutput)


def logReconstruct(toCall: keras.Operation, call_args: list, call_kwargs: dict):
    print(f"Running >> {toCall.name}")
    print(f"\targs >> {call_args}")
    print(f"\tkwargs >> {call_kwargs}")
    print()


def unpack_args(args, producedOutputs):
    op_args = []
    for arg in args:
        if arg is None:
            continue

        if isinstance(arg, keras.KerasTensor):
            hist = arg._keras_history
            prevOpName, tensorIndex = hist.operation.name, hist.tensor_index
            prevOpOutputs = producedOutputs[prevOpName]
            op_args.append(prevOpOutputs[tensorIndex])
        elif isinstance(arg, list):
            op_args.append(unpack_args(arg, producedOutputs))
        else:
            op_args.append(arg)
    return op_args


def unpack_kwargs(kwargs, producedOutputs):
    op_kwargs = {}
    for key in kwargs.keys():
        arg = kwargs[key]
        if arg is None:
            continue

        if isinstance(arg, keras.KerasTensor):
            hist = arg._keras_history
            prevOpName, tensorIndex = hist.operation.name, hist.tensor_index
            prevOpOutputs = producedOutputs[prevOpName]
            op_kwargs[key] = prevOpOutputs[tensorIndex]
        elif isinstance(arg, list):
            op_kwargs[key] = unpack_args(arg, producedOutputs)
        else:
            op_kwargs[key] = arg
    return op_kwargs


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


def findArguments(
    operation: keras.Operation, allSubModels: list[keras.Model]
) -> tuple[list, dict]:
    if isinstance(operation, keras.Model):
        ## It is a sub model
        ## We change the sub model with an Identity Layer
        ## returning the same output as the sub model itself
        return [operation.outputs], {}
    elif isinstance(operation, keras.layers.InputLayer):
        ## It is input layer of sub model
        ## We chnage it with an Identity layer returning
        ## the same output as the sub model
        opSubModel: keras.Model = None
        inputIdx: int = None
        for subMod in allSubModels:
            subModInputs: list[str] = Utils.getInputLayersNames(subMod)
            if operation.name in subModInputs:
                inputIdx = subModInputs.index(operation.name)
                opSubModel = subMod
                break

        ## TODO >> Check this if is enough general
        for argElem in opSubModel._inbound_nodes[0].arguments.args:
            if isinstance(argElem, list):
                return [argElem[inputIdx]], {}
            else:
                return [argElem], {}

    else:
        ## Simple operation
        ## Return its args
        return (
            operation._inbound_nodes[0].arguments.args,
            operation._inbound_nodes[0].arguments.kwargs,
        )
