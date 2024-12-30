import keras
from keras.src.ops.node import KerasHistory, Node
from Manipulation import Utils


class NodeKey:
    def __init__(self, modelKey, opName: str, nodeIdx: int):
        modelKey: NodeKey | None
        keyTuple: tuple = (opName, nodeIdx)
        if modelKey is not None:
            keyTuple: tuple = modelKey.key + keyTuple

        self.key: tuple = keyTuple

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, value):
        return isinstance(value, NodeKey) and self.key == value.key

    def getOpName(self):
        return self.key[-2]

    def getOpIdx(self):
        return self.key[-1]

    def __str__(self) -> str:
        return str(self.key)

    def __repr__(self):
        return str(self.key)

    def format(self) -> str:
        form: str = ""
        for idx, elem in enumerate(self.key):
            if isinstance(elem, str):
                form += f"{elem}"
            else:
                form += f".c{elem}"
                if idx != len(self.key) - 1:
                    form += ">"
        return form


class NodeWrapper:
    def __init__(self, node: Node, nodeIdx: int, model: keras.Model, modelKey: NodeKey):
        self.node: Node = node
        self.nodeIdx: int = nodeIdx
        self.ownerModel: keras.Model = model
        self.ownerModelKey: NodeKey = modelKey

        self.nodeKey: NodeKey = NodeKey(modelKey, node.operation.name, nodeIdx)

    def getId(self) -> NodeKey:
        return self.nodeKey

    def getOwnerModel(self) -> keras.Model:
        return self.ownerModel

    def getOwnerModelKey(self) -> NodeKey:
        return self.ownerModelKey

    def belongsToModel(self, modelKey: NodeKey) -> bool:
        return self.ownerModelKey == modelKey

    def getNodeIdx(self) -> int:
        return self.nodeIdx

    def isKerasModel(self) -> bool:
        nodeOp: keras.Operation = self.node.operation
        return isinstance(nodeOp, keras.Model)

    def isInput(self) -> bool:
        return self.node.is_input

    def isOutput(self) -> bool:
        ownerModelOutput: list[keras.KerasTensor] = Utils.convertToList(
            self.ownerModel.output
        )
        outputHistories: list[KerasHistory] = [
            tens._keras_history for tens in ownerModelOutput
        ]
        prevNodes: list[Node] = [
            hist.operation._inbound_nodes[hist.node_index] for hist in outputHistories
        ]

        return self.node in prevNodes

    def getOperation(self) -> keras.Operation:
        return self.node.operation

    def getOperationOutput(self) -> list[keras.KerasTensor]:
        return self.node.output_tensors

    def getSubModelOuptut(self) -> list[keras.KerasTensor]:
        ## We are returning the output of the sub model represented by this node
        return self.node.operation.outputs

    def getOperationInput(self) -> list[keras.KerasTensor]:
        return self.node.input_tensors

    def getNodeArgs(self) -> tuple:
        return self.node.arguments.args

    def getNodeKwargs(self) -> dict:
        return self.node.arguments.kwargs


class NodePool:
    def __init__(self, model: keras.Model):
        self.perKeyDict: dict[NodeKey, NodeWrapper] = {}
        self.depthSortedKeys: list[NodeKey] = []

        ## Assuming the key of the main model is None as it cannot be reused inside itself
        self.initNodePool(model, None)

    def getAllKeys(self) -> list[NodeKey]:
        return list(self.perKeyDict.keys())

    def getAllNodes(self) -> list[NodeWrapper]:
        return list(self.perKeyDict.values())

    def findModelNodes(model: keras.Model) -> set[Node]:
        nodeSet: set[Node] = set()

        outputs: list[keras.KerasTensor] = model.outputs
        outputHistories: list[KerasHistory] = [out._keras_history for out in outputs]
        nodeQueue: list[Node] = [
            hist.operation._inbound_nodes[hist.node_index] for hist in outputHistories
        ]

        while nodeQueue:
            currNode: Node = nodeQueue.pop()
            nodeSet.add(currNode)

            for parentNode in currNode.parent_nodes:
                if parentNode not in nodeSet and parentNode not in nodeQueue:
                    nodeQueue.append(parentNode)

        return nodeSet

    def initNodePool(self, model: keras.Model, modelKey: NodeKey | None) -> None:
        opsQueue: list[keras.Operation] = Utils.getModelOperations(model)

        modelNodes: set[Node] = NodePool.findModelNodes(model)

        while opsQueue:
            currOp: keras.Operation = opsQueue.pop(0)
            addedKeys: list[NodeKey] = self.addNodesFromOperation(
                currOp, model, modelKey, modelNodes
            )
            self.depthSortedKeys.extend(addedKeys)
            for key in addedKeys:
                nodeWrap: NodeWrapper = self.getNodeFromKey(key)
                if nodeWrap.isKerasModel():
                    ## It is a sub Model --> Getting its Nodes
                    self.initNodePool(model=nodeWrap.getOperation(), modelKey=key)

    def findModelNodesKeys(self, modelKey: NodeKey):
        modelNodes: list[NodeKey] = []
        for node in self.getAllNodes():
            if node.belongsToModel(None):
                modelNodes.append(node.getId())

        return modelNodes

    def addNodesFromOperation(
        self,
        operation: keras.Operation,
        ownerModel: keras.Model,
        ownerModelKey: NodeKey,
        ownerModelNodes: set[Node],
    ) -> list[NodeKey]:

        addedKeys: list[NodeKey] = []
        for nodeIdx, node in enumerate(operation._inbound_nodes):
            if node in ownerModelNodes:
                newOpWrap: NodeWrapper = NodeWrapper(
                    node, nodeIdx, ownerModel, ownerModelKey
                )
                newKey: NodeKey = newOpWrap.getId()
                if newKey not in self.perKeyDict:
                    addedKeys.append(newKey)
                    self.perKeyDict[newKey] = newOpWrap

        return addedKeys

    def getNodeFromKey(self, key: NodeKey) -> NodeWrapper:
        wrap: NodeWrapper = self.perKeyDict.get(key, None)
        if wrap is None:
            raise KeyError(f"Provided Key {key} Is Not Valid")
        return wrap

    def findInputNodesKeys(self, modelKey: NodeKey) -> list[NodeKey]:
        keyList: list[NodeKey] = []
        for key in self.perKeyDict.keys():
            nodeWrap: NodeWrapper = self.perKeyDict[key]
            if nodeWrap.isInput() and nodeWrap.belongsToModel(modelKey=modelKey):
                keyList.append(key)
        return keyList

    def findOutputNodesKeys(self, modelKey: NodeKey) -> list[NodeKey]:
        keyList: list[NodeKey] = []
        for key in self.perKeyDict.keys():
            nodeWrap: NodeWrapper = self.perKeyDict[key]
            if nodeWrap.isOutput() and nodeWrap.belongsToModel(modelKey=modelKey):
                keyList.append(key)

        return keyList

    def getDepthSortedKeys(self) -> list[NodeKey]:
        return self.depthSortedKeys


class PrevFinder:
    def __init__(self):
        pass

    def getPrevsForNormalNode(
        nodeWrap: NodeWrapper, nodePool: NodePool
    ) -> list[NodeWrapper]:
        modelKey: NodeKey = nodeWrap.getOwnerModelKey()
        inputTensors: list[Node] = nodeWrap.getOperationInput()
        wrapList: list[NodeWrapper] = []
        for inputTensor in inputTensors:
            hist: KerasHistory = inputTensor._keras_history
            prevOpName, nodeIdx = hist.operation.name, hist.node_index

            prevNodeKey: NodeKey = NodeKey(modelKey, prevOpName, nodeIdx)
            prevNodeWrap: Node = nodePool.getNodeFromKey(prevNodeKey)
            wrapList.append(prevNodeWrap)

        return wrapList

    def getPrevsForInputNode(
        nodeWrap: NodeWrapper, nodePool: NodePool
    ) -> list[NodeWrapper]:
        ownerModelKey: NodeKey = nodeWrap.getOwnerModelKey()
        ownerModelWrap: NodeWrapper = nodePool.getNodeFromKey(ownerModelKey)

        modelNodeInputTensors: list[keras.KerasTensor] = (
            ownerModelWrap.getOperationInput()
        )
        histories: list[KerasHistory] = [
            inpTens._keras_history for inpTens in modelNodeInputTensors
        ]
        keys: list[NodeKey] = [
            NodeKey(
                ownerModelWrap.getOwnerModelKey(), hist.operation.name, hist.node_index
            )
            for hist in histories
        ]

        modelInputLayerNames: list[str] = Utils.getModelInputLayersNames(
            ownerModelWrap.getOperation()
        )

        prevNodes: list[NodeWrapper] = []
        for idx, name in enumerate(modelInputLayerNames):
            if name == nodeWrap.getOperation().name:
                key: NodeKey = keys[idx]
                prevNodes.append(nodePool.getNodeFromKey(key))
        return prevNodes

    def getPrevsForModelNode(
        nodeWrap: NodeWrapper, nodePool: NodePool
    ) -> list[NodeWrapper]:

        subModOutTensors: list[keras.KerasTensor] = nodeWrap.getSubModelOuptut()
        histories: list[KerasHistory] = [
            outTens._keras_history for outTens in subModOutTensors
        ]
        outNodeKeys: list[NodeKey] = [
            NodeKey(nodeWrap.getId(), hist.operation.name, hist.node_index)
            for hist in histories
        ]

        wrapList: list[NodeWrapper] = []
        for outNodeKey in outNodeKeys:
            wrapList.append(nodePool.getNodeFromKey(outNodeKey))

        return wrapList
