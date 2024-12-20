import keras
from Manipulation import PathGenerator, Utils
from Manipulation.OperationWrapper import OperationWrapper


class ModelGraph:

    def __init__(self, model: keras.Model):
        self.model: keras.Model = model

        ## InputLayerPath --> List of elems like modelName/input_layer_name
        self.inputOpsList: list[str] = self.findInputPaths(model)

        ## InputLayerPath --> List of elems like modelName/output_layer_name
        self.outputOpsList: list[str] = self.findOutputPaths(model)

        # ## Dict Mapping operationPath to its OperationInfo
        self.allOpsDict: dict[str, OperationWrapper] = self.findAllOpsDict(
            model, model.name
        )

        # ## Dict Mapping operationPath to its followers
        self.nextOpsDict: dict[str, set[str]] = self.findNextConns(self.allOpsDict)

        # ## Dict Mapping operationPath to its predecessors
        self.prevOpsDict: dict[str, set[str]] = self.findPrevConns()

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

    def findAllOpsDict(
        self, model: keras.Model, path: str
    ) -> dict[str, OperationWrapper]:
        allOpsDict: dict[str, OperationWrapper] = {}

        opsQueue: list[keras.Operation] = Utils.getModelOperations(model)

        while opsQueue:
            currOp: keras.Operation = opsQueue.pop(0)
            currOpPath = PathGenerator.generatePath(path, currOp.name)
            currOpWrap: OperationWrapper = OperationWrapper(currOp, model, currOpPath)

            if currOpWrap.isKerasModel():
                ## It is a sub Model --> Getting its operations
                subAllOps: dict[str, keras.Operation] = self.findAllOpsDict(
                    currOp, currOpPath
                )
                allOpsDict.update(subAllOps)

            allOpsDict[currOpPath] = currOpWrap

        return allOpsDict

    def findSubModelOps(
        self, model: keras.Model, allOpsDict: dict[str, OperationWrapper]
    ) -> dict[str, OperationWrapper]:
        subAllOpsDict: dict[str, OperationWrapper] = {}
        for key in allOpsDict.keys():
            if allOpsDict[key].belongsToModel(model):
                subAllOpsDict[key] = allOpsDict[key]

        return subAllOpsDict

    def findNextConns(self, allOpsDict: dict[str, OperationWrapper]):
        nextOpsDict: dict[str, list[str]] = {}
        opsQueue = list(allOpsDict.keys())

        while opsQueue:
            currOpName: str = opsQueue.pop()
            currOpWrap: OperationWrapper = self.allOpsDict[currOpName]

            nextOpsDict.setdefault(currOpName, [])

            if currOpWrap.isKerasModel():
                ## It is a SubModel
                print(f"Sub Model > {currOpWrap.getOp().name}")

                ## Find Sub Models Ops
                subAllOpsDict: dict[str, OperationWrapper] = self.findSubModelOps(
                    currOpWrap.getOp(), allOpsDict
                )
                ## Find Internal Connections
                subNextConns: dict[str, set[str]] = self.findNextConns(subAllOpsDict)

                ## If a layer is an output layer --> Connect it to the name of the curr layer
                ## If a layer is on input layer --> Handled correctly (we are looking for nexts nodes)
                for subOpName in subNextConns.keys():
                    subOpWrap: OperationWrapper = allOpsDict[subOpName]
                    nextOpsDict.setdefault(subOpName, [])

                    nextOpsDict[subOpName].extend(subNextConns[subOpName])

                    if subOpWrap.isOutputOp():
                        nextOpsDict[subOpName].append(currOpName)

            currOpNextOpsNames: list[keras.Operation] = currOpWrap.getNextOpsPaths()
            for nextOpName in currOpNextOpsNames:
                nextOpWrap: OperationWrapper = allOpsDict.get(nextOpName, None)

                ## TODO CHECK THIS!!!
                if nextOpWrap is None :
                    continue

                if not nextOpWrap.isKerasModel():
                    ## The next node is not a SubModel
                    ## The next is just the the next operation we are iterating on
                    nextOpsDict[currOpName].append(nextOpName)
                else:
                    ## The next node is a SubModel
                    ## Need to find the input layer corresponding as next for the current node
                    correspondingInputLayers: list[str] = (
                        self.findSubModelCorrespondingInputLayer(nextOpWrap, currOpWrap)
                    )
                    nextOpsDict[currOpName].extend(correspondingInputLayers)

        convNextOpsDict: dict[str, set[str]] = {}
        for key in nextOpsDict.keys():
            convNextOpsDict[key] = set(nextOpsDict[key])

        return convNextOpsDict

    def findPrevConns(self):
        prevOpsDict: dict[str, set[str]] = {}
        for opName in self.nextOpsDict.keys():
            prevConnections = set()
            for otherOpName in self.nextOpsDict.keys():
                otherOpNexts = self.nextOpsDict[otherOpName]
                if opName != otherOpName:
                    for elem in otherOpNexts:
                        if elem == opName:
                            prevConnections.add(otherOpName)
            prevOpsDict[opName] = prevConnections
        return prevOpsDict

    def findSubModelCorrespondingInputLayer(
        self, subModelWrap: OperationWrapper, currOpWrap: OperationWrapper
    ) -> list[str]:
        ## Getting inputs as received by sub model layer
        layerInputsPrevNames: list[str] = subModelWrap.getPrevLayersNames()

        ## Getting inputs as received by sub model
        subModelInputLayers = subModelWrap.getSubModelInputNames()

        ## Find input layers corresponding to this operation
        correspondingLayers = []
        for idx, elem in enumerate(layerInputsPrevNames):
            if elem == currOpWrap.getName():
                correspondingLayers.append(subModelInputLayers[idx])
        return correspondingLayers
