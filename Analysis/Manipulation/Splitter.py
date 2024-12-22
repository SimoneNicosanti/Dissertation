import keras
from Manipulation import Reconstructor, Utils
from Manipulation.ModelGraph import ModelGraph
from Manipulation.NodeWrapper import NodeKey, NodePool, NodeWrapper


def findPrevNodeNeededOutput(currNode: NodeWrapper, prevNode: NodeWrapper) -> list[int]:
    inputTensors = Utils.convertToList(currNode.getOperationInput())

    neededTensorIdxs = []

    for inputTensor in inputTensors:
        inputHist = inputTensor._keras_history
        originNodeName, nodeIndex, tensorIndex = (
            inputHist.operation.name,
            inputHist.node_index,
            inputHist.tensor_index,
        )

        ## Assuming the model unnested, the source node will be in the same model
        originKey: NodeKey = NodeKey(
            currNode.getOwnerModelKey(), originNodeName, nodeIndex
        )
        if originKey == prevNode.getId():
            neededTensorIdxs.append(tensorIndex)

    return neededTensorIdxs


def findInputOpsList(
    subNodeKeys: list[NodeKey],
    nodePool: NodePool,
    prevOpsDict: dict[NodeKey, set[NodeKey]],
) -> dict[NodeKey, set[int]]:

    inputOpsDict: dict[NodeKey, list[int]] = {}
    for nodeKey in subNodeKeys:
        prevNodeKeys: set[NodeKey] = prevOpsDict[nodeKey]
        if len(prevNodeKeys) == 0:
            ## Input Layer
            inputOpsDict.setdefault(nodeKey, [])
            inputOpsDict[nodeKey].extend([0])
        else:
            for prevNodeKey in prevNodeKeys:
                if prevNodeKey not in subNodeKeys:
                    inputOpsDict.setdefault(prevNodeKey, [])
                    tensorIdxs = findPrevNodeNeededOutput(
                        currNode=nodePool.getNodeFromKey(nodeKey),
                        prevNode=nodePool.getNodeFromKey(prevNodeKey),
                    )
                    inputOpsDict[prevNodeKey].extend(tensorIdxs)

    return {nodeKey: set(inputOpsDict[nodeKey]) for nodeKey in inputOpsDict.keys()}


def findOutputOpsList(
    subModelKeys: list[NodeKey],
    nodePool: NodePool,
    nextOpsDict: dict[NodeKey, set[NodeKey]],
    modelOutputKeys: list[NodeKey],
) -> dict[NodeKey, set[int]]:

    outputOpsDict: dict[NodeKey, list[int]] = {}
    for nodeKey in subModelKeys:
        nextNodeKeys: set[NodeKey] = nextOpsDict[nodeKey]

        if nodeKey in modelOutputKeys or len(nextNodeKeys) == 0:
            ## This is an output layer of the model
            outputList = Utils.convertToList(
                nodePool.getNodeFromKey(nodeKey).getOperationOutput()
            )
            outputOpsDict.setdefault(nodeKey, [])
            outputOpsDict[nodeKey].extend([x for x in range(0, len(outputList))])
        else:
            for nextOpKey in nextNodeKeys:
                if nextOpKey not in subModelKeys:
                    outputOpsDict.setdefault(nodeKey, [])
                    tensorIdxs = findPrevNodeNeededOutput(
                        nodePool.getNodeFromKey(nextOpKey),
                        nodePool.getNodeFromKey(nodeKey),
                    )
                    outputOpsDict[nodeKey].extend(tensorIdxs)
    return {nodeKey: set(outputOpsDict[nodeKey]) for nodeKey in outputOpsDict.keys()}


def buildSubModel(
    subNodeKeys: list[NodeKey],
    nodePool: NodePool,
    prevOpsDict: dict[NodeKey, set[NodeKey]],
    nextOpsDict: dict[NodeKey, set[NodeKey]],
    outputLayerKeys: list[NodeKey],
    subModIdx: int,
) -> keras.Model:

    print(f"Building Sub Model Of Index {subModIdx}")
    inputOpsDict: dict[NodeKey, set[int]] = findInputOpsList(
        subNodeKeys, nodePool, prevOpsDict
    )
    outputOpsDict: dict[NodeKey, set[int]] = findOutputOpsList(
        subNodeKeys, nodePool, nextOpsDict, outputLayerKeys
    )

    ## Assuming Unnested Model
    subModel = Reconstructor.reconstructModel(
        subNodeKeys,
        nodePool,
        prevOpsDict,
        inputOpsDict,
        outputOpsDict,
    )

    return subModel


def modelSplit(model: keras.Model, maxLayerNum: int) -> list[keras.Model]:
    modelGraph: ModelGraph = ModelGraph(model)

    allNodeKeys: list[NodeKey] = modelGraph.getDepthSortedKeys()
    nodePool: dict[NodeKey, NodeWrapper] = modelGraph.getNodePool()

    outputLayersKeys: list[NodeKey] = modelGraph.getOutputOpsKeys()

    prevConns, nextConns = modelGraph.prevConns, modelGraph.nextConns
    subModels = []
    modIdx = 0
    for i in range(0, len(allNodeKeys), maxLayerNum):
        subKeys = allNodeKeys[i : min(len(allNodeKeys), i + maxLayerNum)]

        subModel: keras.Model = buildSubModel(
            subKeys,
            nodePool,
            prevConns,
            nextConns,
            outputLayersKeys,
            modIdx,
        )

        subModels.append(subModel)
        modIdx += 1

    return subModels
