import keras
from Manipulation import Reconstructor
from Manipulation.ModelGraph import ModelGraph

from Manipulation.NodeWrapper import NodePool, NodeWrapper, NodeKey


def unnestModel(model: keras.Model):
    modelGraph: ModelGraph = ModelGraph(model)

    nodePool : NodePool = modelGraph.nodePool

    prevOpsDict: dict[NodeKey, set[NodeKey]] = modelGraph.prevConns

    inputOpsKeys: list[NodeKey] = modelGraph.inputOpsKeys
    inputOpsDict = {opName: set([0]) for opName in inputOpsKeys}

    outputOpsKeys: list[NodeKey] = modelGraph.outputOpsKeys
    outputOpsDict : dict[NodeKey, set[int]] = {}
    for outKey in outputOpsKeys:
        outWrap: NodeWrapper = nodePool.getNodeFromKey(outKey)
        opOutputList = outWrap.getOperationOutput()
        outputOpsDict[outKey] = set([x for x in range(0, len(opOutputList))])


    newModel = Reconstructor.reconstructModel(
        nodePool.getAllKeys(),
        nodePool,
        prevOpsDict,
        inputOpsDict,
        outputOpsDict,
    )

    return newModel
