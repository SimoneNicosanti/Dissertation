import keras
from Manipulation import Utils
from Manipulation.NodeWrapper import NodeKey, NodePool, NodeWrapper



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
                shape=out.shape[1:],
                dtype=out.dtype,
                name=f"{inpLayerKey.getOpName()}_{idx}",
            )
            producedOutputs[inpLayerKey].append(tempInpLayer.output)

    return producedOutputs


def logReconstruct(nodeKey: NodeKey, call_args: list, call_kwargs: dict):
    print(f"Running >> {nodeKey}")
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

    nodeWrap: NodeWrapper = nodePool.getNodeFromKey(nodeKey)
    toCall: keras.Operation = wrapOperation(nodeWrap)
    args, kwargs = wrap_args_and_kwargs(nodeWrap, nodePool)

    inputGeneratorKey: NodeKey = findInputGeneratorKey(nodeWrap, nodePool)
    call_args = unpack_args(inputGeneratorKey, args, producedOutputs)
    call_kwargs = unpack_kwargs(inputGeneratorKey, kwargs, producedOutputs)
    logReconstruct(nodeKey, call_args, call_kwargs)
    
    opOutput = toCall(*call_args, **call_kwargs)

    producedOutputs[nodeKey] = Utils.convertToList(opOutput)


def wrapOperation(nodeWrap: NodeWrapper) -> keras.Operation:
    ## In order to keep the model original struct, we change both
    ## input layers and layers representing sub models with IdentityLayers
    newOperation: keras.Operation = None
    if nodeWrap.isKerasModel() or nodeWrap.isInput():
        newOperation = keras.layers.Identity(name=nodeWrap.getId().format())
    else:
        newOperation = nodeWrap.getOperation()
    return newOperation


def wrap_args_and_kwargs(nodeWrap: NodeWrapper, nodePool : NodePool) -> tuple[list, dict]:
    if nodeWrap.isKerasModel():
        ## It is a sub model
        ## We change the sub model with an Identity Layer
        ## returning the same output as the sub model itself
        return [nodeWrap.getSubModelOuptut()], {}  ## From SubModel
    elif nodeWrap.isInput():
        ## It is input layer of sub model
        ## We change it with an Identity layer returning
        ## the same input needed by the submodel model
        subModInputs: list[str] = Utils.getModelInputLayersNames(
            nodeWrap.getNodeModel()
        )
        inputIdx: int = subModInputs.index(nodeWrap.getId().getOpName())

        ownerKey : NodeKey = nodeWrap.getOwnerModelKey()
        ownerNode : NodeWrapper = nodePool.getNodeFromKey(ownerKey)
        ## TODO >> Check this if is enough general
        for argElem in ownerNode.getNodeArgs():  ## Super Model of Wrap
            if isinstance(argElem, list):
                return [argElem[inputIdx]], {}
            elif isinstance(argElem, dict):
                return [argElem[nodeWrap.getId().getOpName()]], {}
            else:
                return [argElem], {}
    else:
        ## Simple operation
        ## Return its args
        return (
            nodeWrap.getNodeArgs(),
            nodeWrap.getNodeKwargs(),
        )  ## Same Model as nodeWrap


def findInputGeneratorKey(nodeWrap: NodeWrapper, nodePool: NodePool) -> NodeKey:
    if nodeWrap.isKerasModel():
        ## The placeholder node will receive the output of the sub model output nodes
        ## We have to return the key of the sub model itself
        return nodeWrap.getId()
    elif nodeWrap.isInput():
        ## The placeholder node will receive the output of a layer in the owner of the owner
        return nodePool.getNodeFromKey(nodeWrap.getOwnerModelKey()).getOwnerModelKey()
    else:
        ## Normal Node so the input will be received by a node in the same model
        return nodeWrap.getOwnerModelKey()


def unpack_args(
    inputGeneratorKey: NodeKey,
    args: list,
    producedOutputs: dict[NodeKey, list[keras.KerasTensor]],
):
    op_args = []
    for arg in args:
        if arg is None:
            continue

        if isinstance(arg, keras.KerasTensor):
            hist = arg._keras_history
            prevOpName, nodeIdx, tensorIndex = (
                hist.operation.name,
                hist.node_index,
                hist.tensor_index,
            )
            ## TODO This will not work for multiple reuse of the same node
            prevNodeKey: NodeKey = NodeKey(inputGeneratorKey, prevOpName, nodeIdx)

            prevOpOutputs = producedOutputs[prevNodeKey]
            op_args.append(prevOpOutputs[tensorIndex])
        elif isinstance(arg, list):
            op_args.append(unpack_args(inputGeneratorKey, arg, producedOutputs))
        else:
            op_args.append(arg)
    return op_args


def unpack_kwargs(
    inputGeneratorKey: NodeKey,
    kwargs: dict,
    producedOutputs: dict[NodeKey, list[keras.KerasTensor]],
):
    op_kwargs = {}
    for key in kwargs.keys():
        arg = kwargs[key]
        if arg is None:
            continue

        if isinstance(arg, keras.KerasTensor):
            hist = arg._keras_history
            prevOpName, nodeIdx, tensorIndex = (
                hist.operation.name,
                hist.node_index,
                hist.tensor_index,
            )
            prevNodeKey: NodeKey = NodeKey(inputGeneratorKey, prevOpName, nodeIdx)
            prevOpOutputs = producedOutputs[prevNodeKey]
            op_kwargs[key] = prevOpOutputs[tensorIndex]
        elif isinstance(arg, list):
            op_kwargs[key] = unpack_args(inputGeneratorKey, arg, producedOutputs)
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
            newModelInput[f"{inpLayerKey.getOpName()}_{idx}"] = producedOutputs[
                inpLayerKey
            ][idx]

    ## TODO Check if this handling of multiple outputs of layer is enough
    newModelOutput = {}
    for outLayerKey in outputOpsDict.keys():
        tensorIdxs = outputOpsDict[outLayerKey]
        for idx in tensorIdxs:
            newModelOutput[f"{outLayerKey.getOpName()}_{idx}"] = producedOutputs[
                outLayerKey
            ][idx]

    newModel = keras.Model(inputs=newModelInput, outputs=newModelOutput)

    return newModel
