import keras


def getModelOperations(model):
    if isinstance(model, keras.Sequential):
        hiddenInputLayer = model.layers[0].input._keras_history.operation
        allLayers = [hiddenInputLayer]
        allLayers.extend(model.layers)
        print("All Layers >> ", allLayers)
        return allLayers
    elif isinstance(model, keras.Model):
        return model.operations
    else:
        raise ValueError("The Model Is Not A Valid Instance")


def getModelOutputNames(model):
    if isinstance(model, keras.Sequential):
        return [model.layers[-1].name]
    elif isinstance(model, keras.Model):
        return model.output_names
    else:
        raise ValueError("The Model Is Not A Valid Instance")


def getInputLayersNames(model: keras.Model | keras.Sequential) -> list[str]:
    if isinstance(model, keras.Sequential):
        hiddenLayer = model.layers[0].input._keras_history.operation
        return [hiddenLayer.name]

    inputLayerNames = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            inputLayerNames.append(layer.name)
    return inputLayerNames


def findOpsListRecursively(model: keras.Model) -> list[keras.Operation]:
    allOpsList: list[keras.Operation] = []

    opsQueue: list[keras.Operation] = getModelOperations(model)

    while opsQueue:
        currOp: keras.Operation = opsQueue.pop(0)

        if isinstance(currOp, keras.Model):
            subAllOps: list[keras.Operation] = findOpsListRecursively(currOp)
            allOpsList.extend(subAllOps)

        allOpsList.append(currOp)

    return allOpsList


def findOpsDictRecursively(model: keras.Model) -> dict[str, keras.Operation]:
    allOpsList: list[keras.Operation] = findOpsListRecursively(model)
    allOpsDict: dict[str, keras.Operation] = {op.name: op for op in allOpsList}

    return allOpsDict


def findConnections(model: keras.Model):
    nextOpsDict = findNextConnections(model)
    prevOpsDict = findPrevConnections(model, nextOpsDict)

    return prevOpsDict, nextOpsDict


def findNextConnections(model: keras.Model) -> dict[str, set[str]]:
    nextOpsDict: dict[str, list[str]] = {}
    opsStack = getModelOperations(model)

    while opsStack:
        currOp = opsStack.pop()
        nextOpsDict.setdefault(currOp.name, [])

        if isinstance(currOp, keras.Model):
            ## Find Internal Connections
            subNextConns: dict[str, set[str]] = findNextConnections(currOp)

            ## If a level is an output level --> Connect it to the name of the curr level
            ## If a level is on input level --> Handled correctly (we are looking for nexts nodes)
            for subOpName in subNextConns.keys():
                nextOpsDict.setdefault(subOpName, [])
                nextOpsDict[subOpName].extend(subNextConns[subOpName])

                if subOpName in getModelOutputNames(currOp):
                    nextOpsDict[subOpName].append(currOp.name)

        currOpNextOps = [node.operation for node in currOp._outbound_nodes]
        for nextOp in currOpNextOps:
            if not isinstance(nextOp, keras.Model):
                ## The next is just the the next operation we are iterating on
                nextOpsDict[currOp.name].append(nextOp.name)
            else:
                ## Need to find the input layer corresponding as next for the current node
                correspondingInputLayers = findSubModelCorrespondingInputLayer(
                    nextOp, currOp.name
                )
                nextOpsDict[currOp.name].extend(correspondingInputLayers)

    convNextOpsDict: dict[str, set[str]] = {}
    for key in nextOpsDict.keys():
        convNextOpsDict[key] = set(nextOpsDict[key])

    return convNextOpsDict


def findSubModelCorrespondingInputLayer(
    subModel: keras.Model, currOpName: str
) -> list[str]:
    ## Getting inputs as received by sub model layer
    layerInputs = subModel._inbound_nodes[0].arguments._flat_arguments
    layerInputsPrevNames = [
        inp._keras_history.operation.name
        for inp in layerInputs
        if isinstance(inp, keras.KerasTensor)
    ]

    ## Getting inputs as received by sub model
    subModInputs = subModel.inputs
    subModelInputLayers = [inp._keras_history.operation.name for inp in subModInputs]

    ## Find input layers corresponding to this operation
    correspondingLayers = []
    for idx, elem in enumerate(layerInputsPrevNames):
        if elem == currOpName:
            correspondingLayers.append(subModelInputLayers[idx])
    return correspondingLayers


def convertToList(anyValue):
    if isinstance(anyValue, list):
        return anyValue
    elif isinstance(anyValue, dict):
        return anyValue.values()
    else:
        return [anyValue]


def findPrevConnections(model: keras.Model, nextOpsDict=None):
    if nextOpsDict is None:
        nextOpsDict = findNextConnections(model)

    prevOpsDict: dict[str, set[str]] = {}
    for opName in nextOpsDict.keys():
        prevConnections = set()
        for otherOpName in nextOpsDict.keys():
            otherOpNexts = nextOpsDict[otherOpName]
            if opName != otherOpName:
                for elem in otherOpNexts:
                    if elem == opName:
                        prevConnections.add(otherOpName)
        prevOpsDict[opName] = prevConnections
    return prevOpsDict
