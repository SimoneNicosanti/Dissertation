import keras


def findLayersConnections(model: keras.Model):
    config = model.get_config()
    opsList = config["layers"]
    opsConfigDict = {}
    for opConfig in opsList:
        opName = opConfig["name"]
        opsConfigDict[opName] = opConfig

    layerNames = [layer.name for layer in model.layers]

    prevOpsDict = {}
    nextOpsDict = {}
    for layerName in layerNames:
        prevOpsDict[layerName] = []
        nextOpsDict[layerName] = []

    for layerName in layerNames:
        currConfig = opsConfigDict[layerName]
        prevLayers = findPrevLayersFromConfig(currConfig, opsConfigDict, layerNames)

        prevOpsDict[layerName] = prevLayers

        for prevLayer in prevLayers:
            nextOpsDict[prevLayer].append(layerName)

    return prevOpsDict, nextOpsDict


def modelParse(model: keras.Model):
    prevOpsDict, nextOpsDict = findLayersConnections(model)

    maxLayerNum = 40
    modelIdx = 0
    for i in range(0, len(model.layers), maxLayerNum):
        subLayers = model.layers[i : min(len(model.layers), i + maxLayerNum)]
        subLayersNames = [x.name for x in subLayers]

        subModelInput = buildSubModelInput(
            subLayers, subLayersNames, model, prevOpsDict
        )
        subModelOutput = buildSubModelOutput(
            subLayers, subLayersNames, model, nextOpsDict
        )

        subModel = keras.Model(inputs=subModelInput, outputs=subModelOutput)
        subModel.export(f"/models/SubModel_{modelIdx}")

        modelIdx += 1


def findPrevLayersFromConfig(currConfig, configDict, layerNames):
    prevOpsNames = []
    prevLayers = []
    findHistoryInDepth(currConfig["inbound_nodes"], prevOpsNames)
    for opName in prevOpsNames:
        if opName in layerNames:
            ## It is a valid layer --> Append to prev Layers
            prevLayers.append(opName)
        else:
            ## It is other operation --> Find first valid prev layer
            opConfig = configDict[opName]
            opPrevLayers = findPrevLayersFromConfig(opConfig, configDict, layerNames)
            prevLayers += opPrevLayers

    return prevLayers


def findHistoryInDepth(node, histories: list):
    if isinstance(node, dict):
        ## Check if key is present
        if "keras_history" in node:
            histories.append(node["keras_history"][0])  ## Takes operation name
        else:
            for key in node:
                findHistoryInDepth(node[key], histories)
    elif isinstance(node, list) or isinstance(node, tuple):
        ## Check each elem for the key
        for elem in node:
            findHistoryInDepth(elem, histories)

    return


def buildSubModelInput(subLayers, subLayersNames, model, prevOpsDict):
    subModelInput = {}
    for layer in subLayers:
        if len(prevOpsDict[layer.name]) == 0:
            ## Input Layer
            newInput = keras.Input(
                shape=layer.output.shape[1:],
                tensor=layer.output,
                name=layer.name,
            )
            newInput.name = layer.name
            subModelInput[layer.name] = newInput
        else:
            for prevLayerName in prevOpsDict[layer.name]:
                prevLayerOut = model.get_layer(prevLayerName).output
                # print(prevLayerOut)
                if prevLayerName not in subLayersNames:
                    # newInput = keras.Input(tensor=prevLayerOut)
                    newInput = keras.Input(
                        shape=prevLayerOut.shape[1:],
                        tensor=prevLayerOut,
                        name=prevLayerName,
                    )
                    newInput.name = prevLayerName
                    subModelInput[prevLayerName] = newInput

    return subModelInput


def buildSubModelOutput(subLayers, subLayersNames, model, nextOpsDict):
    subModelOutput = {}
    for layer in subLayers:
        layerOutput = layer.output
        if len(nextOpsDict[layer.name]) == 0:
            ## Output Layer
            subModelOutput[layer.name] = layerOutput
        else:
            for nextLayer in nextOpsDict[layer.name]:
                if nextLayer not in subLayersNames:
                    subModelOutput[layer.name] = layerOutput
    return subModelOutput
