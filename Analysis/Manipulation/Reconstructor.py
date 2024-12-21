import keras
from Manipulation import Utils
from Manipulation.NodeWrapper import NodeWrapper, NodePool, NodeKey


def initProducedOutputs(
    inputOpsDict: dict[NodeKey, set[int]], nodePool: NodePool
) -> dict[NodeKey, list[keras.KerasTensor]]:
    producedOutputs: dict[NodeKey, list[keras.KerasTensor]] = {}
    for inpLayerKey in inputOpsDict.keys():
        producedOutputs.setdefault(inpLayerKey, [])

        currNodeWrap: NodeWrapper = nodePool.getNodeFromKey(inpLayerKey)
        outputList = currNodeWrap.getOperationOutput()
        for idx, out in enumerate(outputList):
            tempInpLayer = keras.layers.InputLayer(
                shape=out.shape[1:], dtype=out.dtype, name=f"{inpLayerKey.getOpName()}_{idx}"
            )
            producedOutputs[inpLayerKey].append(tempInpLayer.output)

    return producedOutputs


def logReconstruct(toCall: keras.Operation, call_args: list, call_kwargs: dict):
    print(f"Running >> {toCall.name}")
    print(f"\targs >> {call_args}")
    print(f"\tkwargs >> {call_kwargs}")
    print()


def runOperation(
    nodeKey: str,
    nodePool: NodePool,
    prevOpsDict: dict[NodeKey, set[NodeKey]],
    producedOutputs: dict[NodeKey, list[keras.KerasTensor]],
):  
    ## Run all needed previous operations
    for prevOpKey in prevOpsDict[nodeKey]:
        if prevOpKey not in producedOutputs:
            runOperation(prevOpKey, nodePool, prevOpsDict, producedOutputs)


    opWrap: NodeWrapper = nodePool.getNodeFromKey(nodeKey)
    toCall: keras.Operation = wrapOperation(opWrap)
    args, kwargs = wrap_args_and_kwargs(opWrap)


    call_args = unpack_args(args, producedOutputs, nodePool)
    call_kwargs = unpack_kwargs(kwargs, producedOutputs, nodePool)
    logReconstruct(toCall, call_args, call_kwargs)

    opOutput = toCall(*call_args, **call_kwargs)

    producedOutputs[nodeKey] = Utils.convertToList(opOutput)

def wrapOperation(nodeWrap : NodeWrapper) -> keras.Operation:
    ## In order to keep the model original struct, we change both
    ## input layers and layers representing sub models with IdentityLayers
    newOperation: keras.Operation = None
    if nodeWrap.isKerasModel() or nodeWrap.isInput():
        newOperation = keras.layers.Identity(name=nodeWrap.getId().getOpName())
    else:
        newOperation = nodeWrap.getOperation()
    return newOperation

def wrap_args_and_kwargs(nodeWrap : NodeWrapper) -> tuple[list, dict]:
    if nodeWrap.isKerasModel():
        ## It is a sub model
        ## We change the sub model with an Identity Layer
        ## returning the same output as the sub model itself
        return [nodeWrap.getSubModelOuptut()], {}
    elif nodeWrap.isInput():
        ## It is input layer of sub model
        ## We change it with an Identity layer returning
        ## the same input needed by the submodel model
        subModInputs: list[str] = Utils.getInputLayersNames(nodeWrap.getNodeModel())
        inputIdx: int = subModInputs.index(nodeWrap.getId().getOpName())

        ## TODO >> Check this if is enough general
        for argElem in nodeWrap.getOwnerModelArgs():
            if isinstance(argElem, list):
                return [argElem[inputIdx]], {}
            else:
                return [argElem], {}
    else:
        ## Simple operation
        ## Return its args
        return (
            nodeWrap.getNodeArgs(),
            nodeWrap.getNodeKwargs(),
        )


def unpack_args(args, producedOutputs : dict[NodeKey, list[keras.KerasTensor]], nodePool : NodePool):
    op_args = []
    for arg in args:
        if arg is None:
            continue

        if isinstance(arg, keras.KerasTensor):
            hist = arg._keras_history
            prevOpName, tensorIndex = hist.operation.name, hist.tensor_index
            ## TODO This will not work for multiple reuse of the same node
            prevNodeWrap : NodeWrapper = nodePool.getNodesFromOpName(prevOpName)[0] 

            prevOpOutputs = producedOutputs[prevNodeWrap.getId()]
            op_args.append(prevOpOutputs[tensorIndex])
        elif isinstance(arg, list):
            op_args.append(unpack_args(arg, producedOutputs, nodePool))
        else:
            op_args.append(arg)
    return op_args


def unpack_kwargs(kwargs, producedOutputs : dict[NodeKey, list[keras.KerasTensor]], nodePool : NodePool):
    op_kwargs = {}
    for key in kwargs.keys():
        arg = kwargs[key]
        if arg is None:
            continue

        if isinstance(arg, keras.KerasTensor):
            hist = arg._keras_history
            prevOpName, tensorIndex = hist.operation.name, hist.tensor_index
            ## TODO This will not work for multiple reuse of the same node
            prevNodeWrap : NodeWrapper = nodePool.getNodesFromOpName(prevOpName)[0]
            prevOpOutputs = producedOutputs[prevNodeWrap.getId()]
            op_kwargs[key] = prevOpOutputs[tensorIndex]
        elif isinstance(arg, list):
            op_kwargs[key] = unpack_args(arg, producedOutputs, nodePool)
        else:
            op_kwargs[key] = arg
    return op_kwargs


def reconstructModel(
    allNodesKeys: list[NodeKey],
    nodePool: NodePool,
    prevOpsDict: dict[NodeKey, set[NodeKey]],
    inputOpsDict: dict[NodeKey, set[int]],
    outputOpsDict: dict[NodeKey, set[int]],
):

    producedOutputs: dict[NodeKey, list[keras.KerasTensor]] = initProducedOutputs(
        inputOpsDict, nodePool
    )

    for nodeKey in allNodesKeys:
        if nodeKey not in producedOutputs:
            runOperation(nodeKey, nodePool, prevOpsDict, producedOutputs)

    newModelInput = {}
    for inpLayerKey in inputOpsDict.keys():
        tensorIdxs = inputOpsDict[inpLayerKey]
        for idx in tensorIdxs:
            newModelInput[f"{inpLayerKey.getOpName()}_{idx}"] = producedOutputs[inpLayerKey][idx]

    ## TODO Check if this handling of multiple outputs of layer is enough
    newModelOutput = {}
    for outLayerKey in outputOpsDict.keys():
        tensorIdxs = outputOpsDict[outLayerKey]
        for idx in tensorIdxs:
            newModelOutput[f"{outLayerKey.getOpName()}_{idx}"] = producedOutputs[outLayerKey][idx]

    newModel = keras.Model(inputs=newModelInput, outputs=newModelOutput)

    return newModel
