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

    def __str__(self) -> str:
        return str(self.key)

    def __repr__(self):
        return str(self.key)

    def format(self) -> str:
        form: str = ""
        for elem in self.key:
            if isinstance(elem, str):
                form += f"{elem}"
            else:
                form += f"[{elem}]"


class NodeWrapper:
    def __init__(self, node: Node, nodeIdx: int, model: keras.Model, modelKey: NodeKey):
        self.node: Node = node
        self.nodeIdx: int = nodeIdx
        self.nodeModel: keras.Model = model
        self.modelKey: NodeKey = modelKey

        self.nodeKey: NodeKey = NodeKey(modelKey, node.operation.name, nodeIdx)

    def getId(self) -> NodeKey:
        return self.nodeKey

    def getNodeModel(self) -> keras.Model:
        return self.nodeModel

    def getOwnerModelKey(self) -> NodeKey:
        return self.modelKey

    def belongsToMainModel(self, model) -> bool:
        return self.modelKey is None

    def getNodeIdx(self) -> int:
        return self.nodeIdx

    def isKerasModel(self) -> bool:
        nodeOp: keras.Operation = self.node.operation
        return isinstance(nodeOp, keras.Model)

    def isInput(self) -> bool:
        return self.node.is_input

    def isOutput(self) -> bool:
        return self.node.operation.name in Utils.getModelOutputLayersNames(
            self.nodeModel
        )

    def getOperation(self) -> keras.Operation:
        return self.node.operation

    def getOperationOutput(self) -> list[keras.KerasTensor]:
        return self.node.output_tensors

    def getSubModelOuptut(self) -> list[keras.KerasTensor]:
        return self.node.operation.outputs

    def getOperationInput(self) -> list[keras.KerasTensor]:
        return self.node.input_tensors

    def getNodeArgs(self) -> tuple:
        return self.node.arguments.args

    def getNodeKwargs(self) -> dict:
        return self.node.arguments.kwargs


class NodePool:
    def __init__(self):
        self.perKeyDict: dict[NodeKey, NodeWrapper] = {}

    def getAllKeys(self) -> list[NodeKey]:
        return list(self.perKeyDict.keys())

    def addNodesFromOperation(
        self, operation: keras.Operation, model: keras.Model, modelKey: NodeKey
    ) -> list[NodeKey]:

        ## We have to find which nodes are in this model!!
        ## Then we can find which of the _inbound_nodes of this op are in the model
        ## Taking the intersection
        modelNodes: list[Node] = []
        for op in Utils.getModelOperations(model):
            op: keras.Operation
            if isinstance(op, keras.layers.InputLayer):
                modelNodes.extend(op._inbound_nodes)
            for node in op._outbound_nodes:
                modelNodes.append(node)

        addedKeys: list[NodeKey] = []
        for nodeIdx, node in enumerate(operation._inbound_nodes):
            if node in modelNodes:
                newOpWrap: NodeWrapper = NodeWrapper(node, nodeIdx, model, modelKey)
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

    def findInputNodesKeys(self, model: keras.Model) -> list[NodeKey]:
        keyList: list[NodeKey] = []
        for key in self.perKeyDict.keys():
            nodeWrap: NodeWrapper = self.perKeyDict[key]
            if nodeWrap.isInput() and nodeWrap.belongsToMainModel(model):
                keyList.append(key)
        return keyList

    def findOutputNodesKeys(self, model: keras.Model) -> list[NodeKey]:
        keyList: list[NodeKey] = []
        for key in self.perKeyDict.keys():
            nodeWrap: NodeWrapper = self.perKeyDict[key]
            if nodeWrap.isOutput() and nodeWrap.belongsToMainModel(model):
                keyList.append(key)

        return keyList


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
