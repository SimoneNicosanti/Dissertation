import keras
from Manipulation import PathGenerator, Utils
from Manipulation.NodeWrapper import NodePool, NodeWrapper, PrevFinder, NodeKey


class ModelGraph:

    def __init__(self, model: keras.Model):
        self.model: keras.Model = model

        # # Dict Mapping operationPath to its OperationInfo
        self.nodePool: NodePool = NodePool()
        self.initNodePool(self.model)

        ## InputLayerPath --> List of elems like modelName/input_layer_name
        self.inputOpsKeys: list[str] = self.nodePool.findInputNodesKeys(model)

        ## InputLayerPath --> List of elems like modelName/output_layer_name
        self.outputOpsKeys: list[str] = self.nodePool.findOutputNodesKeys(model)

        # # ## Dict Mapping operationPath to its followers
        # self.nextOpsDict: dict[str, set[str]] = self.findNextConns(self.allOpsDict)

        # # ## Dict Mapping operationPath to its predecessors
        self.prevConns: dict[NodeKey, set[NodeKey]] = {}
        self.findPrevConns(self.model)

    def getNodePool(self) -> NodePool:
        return self.nodePool

    def findInputPaths(self, model: keras.Model) -> list[str]:
        inputNames: list[str] = Utils.getInputLayersNames(model)
        pathList = []
        for name in inputNames:
            pathList.append(PathGenerator.generatePath(model.name, name))
        return pathList

    def findOutputPaths(self, model: keras.Model) -> list[str]:
        outputNames: list[str] = Utils.getModelOutputNames(model)
        pathList = []
        for name in outputNames:
            pathList.append(PathGenerator.generatePath(model.name, name))
        return pathList

    def initNodePool(self, model: keras.Model) -> None:
        opsQueue: list[keras.Operation] = Utils.getModelOperations(model)

        while opsQueue:
            currOp: keras.Operation = opsQueue.pop(0)
            self.nodePool.addNodesFromOperation(currOp, model)
            oneNodeWrap: NodeWrapper = self.nodePool.getNodesFromOpName(currOp.name)[0]

            if oneNodeWrap.isKerasModel():
                ## It is a sub Model --> Getting its Nodes
                self.initNodePool(oneNodeWrap.getOperation())

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
                if not currNode.belongsToModel(model):
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
