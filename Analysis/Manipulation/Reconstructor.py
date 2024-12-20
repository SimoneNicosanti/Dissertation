import keras
from Manipulation.OperationWrapper import OperationWrapper
from Manipulation import Utils

def initProducedOutputs(inputOpsDict: dict[str, set[int]], allOpsDict : dict[str, OperationWrapper]) -> dict[str, list[keras.KerasTensor]] :
    producedOutputs: dict[str, list[keras.KerasTensor]] = {}
    for inpLayerName in inputOpsDict.keys():
        producedOutputs.setdefault(inpLayerName, [])

        currOpWrap : OperationWrapper = allOpsDict[inpLayerName]
        outputList = Utils.convertToList(currOpWrap.getOpOutput())
        for idx, out in enumerate(outputList):
            tempInpLayer = keras.layers.InputLayer(
                shape=out.shape[1:], dtype=out.dtype, name=f"{inpLayerName}_{idx}"
            )
            producedOutputs[inpLayerName].append(tempInpLayer.output)
    
    return producedOutputs

def logReconstruct(toCall: keras.Operation, call_args: list, call_kwargs: dict):
    print(f"Running >> {toCall.name}")
    print(f"\targs >> {call_args}")
    print(f"\tkwargs >> {call_kwargs}")
    print()

def runOperation(
    opName: str,
    allOpsDict: dict[str, OperationWrapper],
    prevOpsDict: dict[str, set[str]],
    producedOutputs: dict[str, list[keras.KerasTensor]],
):
    ## Run all needed previous operations
    for prevOpName in prevOpsDict[opName]:
        if prevOpName not in producedOutputs:
            runOperation(
                prevOpName, allOpsDict, prevOpsDict, producedOutputs
            )

    opWrap : OperationWrapper = allOpsDict[opName]
    toCall: keras.Operation = opWrap.getCallable()
    args, kwargs = opWrap.getArguments()

    call_args = unpack_args(args, producedOutputs)
    call_kwargs = unpack_kwargs(kwargs, producedOutputs)
    #logReconstruct(toCall, call_args, call_kwargs)

    opOutput = toCall(*call_args, **call_kwargs)

    producedOutputs[opName] = Utils.convertToList(opOutput)


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
    


def reconstructModel(
        allOpsNames: list[str],
        allOpsDict: dict[str, OperationWrapper],
        prevOpsDict: dict[str, set[str]],
        inputOpsDict: dict[str, set[int]],
        outputOpsDict: dict[str, set[int]],
    ) :

    producedOutputs : dict[str, list[keras.KerasTensor]] = initProducedOutputs(inputOpsDict, allOpsDict)

    for opName in allOpsNames:
        if opName not in producedOutputs:
            runOperation(opName, allOpsDict, prevOpsDict, producedOutputs)

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

