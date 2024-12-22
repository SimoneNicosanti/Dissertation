import keras
from Manipulation import Utils
from Manipulation.NodeWrapper import NodeKey, NodePool, NodeWrapper, PrevFinder


class ModelGraph:

    def __init__(self, model: keras.Model):
        self.model: keras.Model = model

        # # Dict Mapping operationPath to its OperationInfo
        self.nodePool: NodePool = NodePool()
        self.depthSortedKeys: list[NodeKey] = []
        self.initNodePool(self.model)

        self.inputOpsKeys: list[NodeKey] = self.nodePool.findInputNodesKeys(model)

        self.outputOpsKeys: list[NodeKey] = self.nodePool.findOutputNodesKeys(model)

        self.prevConns: dict[NodeKey, set[NodeKey]] = {}
        self.findPrevConns(self.model)

        self.nextConns: dict[NodeKey, set[NodeKey]] = self.findNextConns(self.prevConns)

    def getNodePool(self) -> NodePool:
        return self.nodePool

    def getInputOpsKeys(self) -> list[NodeKey]:
        return self.inputOpsKeys

    def getOutputOpsKeys(self) -> list[NodeKey]:
        return self.outputOpsKeys

    def initNodePool(self, model: keras.Model, modelKey: NodeKey = None) -> None:
        opsQueue: list[keras.Operation] = Utils.getModelOperations(model)

        while opsQueue:
            currOp: keras.Operation = opsQueue.pop(0)
            addedKeys: list[NodeKey] = self.nodePool.addNodesFromOperation(
                currOp, model, modelKey
            )
            self.depthSortedKeys.extend(addedKeys)
            for key in addedKeys:
                nodeWrap: NodeWrapper = self.nodePool.getNodeFromKey(key)
                if nodeWrap.isKerasModel():
                    ## It is a sub Model --> Getting its Nodes
                    self.initNodePool(model=nodeWrap.getOperation(), modelKey=key)

    def getDepthSortedKeys(self) -> list[NodeKey]:
        return self.depthSortedKeys

    def findPrevConns(self, model: keras.Model) -> None:
        nodeKeyQueue: list[NodeKey] = self.nodePool.getAllKeys()

        while nodeKeyQueue:
            currKey: NodeKey = nodeKeyQueue.pop()
            currNode: NodeWrapper = self.nodePool.getNodeFromKey(currKey)
            currNodePrevs: set[NodeKey] = set()

            if currNode.isKerasModel():
                ## Handle Model Case
                ## Predecessors for this node have to be set as the sub model output nodes
                parentWrapList: list[NodeWrapper] = PrevFinder.getPrevsForModelNode(
                    currNode, self.nodePool
                )

            elif currNode.isInput():
                ## Handle Input Case
                if not currNode.belongsToMainModel(model):
                    ## Sub Model Input Layer --> Then its predecessors will be the nodes giving inputs to the sub model
                    parentWrapList: list[NodeWrapper] = PrevFinder.getPrevsForInputNode(
                        currNode, self.nodePool
                    )
                else:
                    ## Main Model Input Layer --> Ok
                    parentWrapList = []

            else:
                ## Just a Common Op
                parentWrapList: list[NodeWrapper] = PrevFinder.getPrevsForNormalNode(
                    currNode, self.nodePool
                )

            for parentWrap in parentWrapList:
                currNodePrevs.add(parentWrap.getId())

            self.prevConns[currKey] = currNodePrevs

    def findNextConns(
        self, prevConns: dict[NodeKey, set[NodeKey]]
    ) -> dict[NodeKey, set[NodeKey]]:
        nextConns: dict[NodeKey, set[NodeKey]] = {}
        for nodeKey in self.depthSortedKeys:
            nextConns.setdefault(nodeKey, set())
        for dest, sources in prevConns.items():
            for src in sources:
                nextConns[src].add(dest)

        return nextConns

    def printConnections(self) -> None:
        for key in self.prevConns.keys():
            print(f"{key}")
            for prevKey in self.prevConns[key]:
                print(f"\t{prevKey}")
