import keras
from keras.src.ops.node import KerasHistory, Node
from Manipulation import Utils

class NodeKey :
    def __init__(self, key : tuple) :
        self.key : tuple = key

    def __hash__(self) :
        return hash(self.key)
    
    def __eq__(self, value):
        return isinstance(value, NodeKey) and self.key == value.key
    
    def getOpName(self) :
        return self.key[-2]

    def __str__(self) -> str :
        return str(self.key)


class NodeWrapper:
    def __init__(self, node: Node, nodeIdx: int, model: keras.Model, nodeKey : NodeKey):
        self.node: Node = node
        self.nodeIdx: int = nodeIdx
        self.nodeModel: keras.Model = model
        self.nodeKey : NodeKey = nodeKey

    def getId(self) -> NodeKey:
        return self.nodeKey

    def getNodeModel(self) -> keras.Model:
        return self.nodeModel

    def belongsToModel(self, model) -> bool:
        return model.name == self.nodeModel.name

    def getNodeIdx(self) -> int:
        return self.nodeIdx

    def isKerasModel(self) -> bool:
        nodeOp: keras.Operation = self.node.operation
        return isinstance(nodeOp, keras.Model)

    def isInput(self) -> bool:
        return self.node.is_input
    
    def isOutput(self) -> bool :
        return self.node.operation.name in Utils.getModelOutputNames(self.nodeModel)

    def getOperation(self) -> keras.Operation:
        return self.node.operation
    
    def getOperationOutput(self) -> list[keras.KerasTensor] :
        return self.node.output_tensors
    
    def getSubModelOuptut(self) -> list[keras.KerasTensor] :
        return self.node.operation.outputs
    
    def getOperationInput(self) -> list[keras.KerasTensor] :
        return self.node.input_tensors
    
    def getNodeArgs(self) -> tuple :
        return self.node.arguments.args
    
    def getNodeKwargs(self) -> dict :
        return self.node.arguments.kwargs

    def getOwnerModelArgs(self) -> dict :
        ## TODO Will not work for reuse of the layer
        return self.nodeModel._inbound_nodes[0].arguments.args


class NodePool:
    def __init__(self):
        self.perIdDict: dict[NodeKey, NodeWrapper] = {}
        self.perNameDict: dict[str, list[NodeWrapper]] = {}

    def getAllKeys(self) -> list[NodeKey]:
        return list(self.perIdDict.keys())

    def addNodesFromOperation(
        self, operation: keras.Operation, model: keras.Model
    ) -> None:
        nodeWrapList: list[NodeWrapper] = []

        for nodeIdx, node in enumerate(operation._inbound_nodes):
            newKey: NodeKey = self.buildKey(node, nodeIdx)
            if newKey not in self.perIdDict:
                ## TODO This cannot handle the reuse case
                ## To do that should build unique keys for reused layers
                newOpWrap: NodeWrapper = NodeWrapper(node, nodeIdx, model, newKey)
                self.perIdDict[newKey] = newOpWrap
                nodeWrapList.append(newOpWrap)

        if operation.name not in self.perNameDict:
            self.perNameDict[operation.name] = nodeWrapList

    def buildKey(self, node: Node, nodeIdx: int) -> NodeKey:
        keyTuple : tuple = (node.operation.name, nodeIdx)
        return NodeKey(keyTuple)

    def getNodeFromKey(self, key: NodeKey) -> NodeWrapper:
        wrap: NodeWrapper = self.perIdDict.get(key, None)
        if wrap is None:
            raise KeyError(f"Provided Key {key} Is Not Valid")
        return wrap

    def getNodesFromOpName(self, opName: str):
        wrapList: list[NodeWrapper] = self.perNameDict.get(opName, None)
        if wrapList is None:
            raise KeyError(f"Provided Operation Name {opName} Is Not Valid")
        return wrapList

    def findInputNodesKeys(self, model : keras.Model) -> list[NodeKey] :
        keyList : list[NodeKey] = []
        for key in self.perIdDict.keys() :
            nodeWrap : NodeWrapper = self.perIdDict[key]
            if nodeWrap.isInput() and nodeWrap.belongsToModel(model) :
                keyList.append(key)
        return keyList
    
    def findOutputNodesKeys(self, model : keras.Model) -> list[NodeKey] :
        keyList : list[NodeKey] = []
        for key in self.perIdDict.keys() :
            nodeWrap : NodeWrapper = self.perIdDict[key]
            if nodeWrap.isOutput() and nodeWrap.belongsToModel(model) :
                keyList.append(key)
        
        return keyList


class PrevFinder:
    def __init__(self):
        pass

    def getPrevsForNormalNode(
        nodeWrap: NodeWrapper, nodePool: NodePool
    ) -> list[NodeWrapper]:
        inputTensors: list[Node] = nodeWrap.node.input_tensors
        wrapList: list[NodeWrapper] = []
        for inputTensor in inputTensors:
            hist: KerasHistory = inputTensor._keras_history
            prevOp, nodeIdx = hist.operation, hist.node_index
            prevNode: Node = prevOp._inbound_nodes[nodeIdx]
            key: tuple = nodePool.buildKey(prevNode, nodeIdx)

            prevNodeWrap: NodeWrapper = nodePool.perIdDict[key]
            wrapList.append(prevNodeWrap)

        return wrapList

    ## TODO This method would not work properly for multiple replicas of the sub model
    def getPrevsForInputNode(
        nodeWrap: NodeWrapper, nodePool: NodePool
    ) -> list[NodeWrapper]:
        ownerModel: keras.Model = nodeWrap.getNodeModel()

        modelNodeInputTensors: list[keras.KerasTensor] = ownerModel._inbound_nodes[
            0
        ].input_tensors
        histories: list[KerasHistory] = [
            inpTens._keras_history for inpTens in modelNodeInputTensors
        ]
        keys: list[tuple] = [
            nodePool.buildKey(
                hist.operation._inbound_nodes[hist.node_index], hist.node_index
            )
            for hist in histories
        ]

        modelInputLayerNames: list[str] = Utils.getInputLayersNames(ownerModel)

        prevNodes: list[NodeWrapper] = []
        for idx, name in enumerate(modelInputLayerNames):
            if name == nodeWrap.getOperation().name:
                key: tuple = keys[idx]
                prevNodes.append(nodePool.getNodeFromKey(key))
        return prevNodes

    def getPrevsForModelNode(
        nodeWrap: NodeWrapper, nodePool: NodePool
    ) -> list[NodeWrapper]:
        subModOutLayerName: list[str] = Utils.getModelOutputNames(
            nodeWrap.getOperation()
        )

        wrapList: list[NodeWrapper] = []
        for name in subModOutLayerName:
            subOutNodes: list[NodeWrapper] = nodePool.getNodesFromOpName(name)
            wrapList.extend(subOutNodes)
        return wrapList