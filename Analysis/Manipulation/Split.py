import keras
from Manipulation import Reconstruct, Utils


def findPrevOperationNeededOutput(
    currOp: keras.Operation, prevOp: keras.Operation
) -> list[int]:
    currOpInputList = Utils.convertToList(currOp.input)

    neededTensorIdxs = []

    for currOpInput in currOpInputList:
        inputHist = currOpInput._keras_history
        inputOriginOpName, tensorIndex = (
            inputHist.operation.name,
            inputHist.tensor_index,
        )

        if inputOriginOpName == prevOp.name:
            neededTensorIdxs.append(tensorIndex)

    return neededTensorIdxs


def findInputOpsList(
    subOpsNames: list[str],
    allOpsDict: dict[str, keras.KerasTensor],
    prevOpsDict: dict[str, set[str]],
) -> dict[str, set[int]]:

    inputOpsDict: dict[str, list[int]] = {}
    for opName in subOpsNames:
        prevOpsNameList: list[str] = prevOpsDict[opName]
        if len(prevOpsNameList) == 0:
            ## Input Layer
            inputOpsDict.setdefault(opName, [])
            inputOpsDict[opName].extend([0])
        else:
            for prevOpName in prevOpsNameList:
                if prevOpName not in subOpsNames:
                    inputOpsDict.setdefault(prevOpName, [])
                    tensorIdxs = findPrevOperationNeededOutput(
                        allOpsDict[opName], allOpsDict[prevOpName]
                    )
                    inputOpsDict[prevOpName].extend(tensorIdxs)

    return {opName: set(inputOpsDict[opName]) for opName in inputOpsDict.keys()}


def findOutputOpsList(
    subOpsNames: list[str],
    allOpsDict: dict[str, keras.Operation],
    nextOpsDict: dict[str, set[str]],
    modelOutputNames: list[str],
) -> dict[str, set[int]]:
    outputOpsDict: dict[str, list[int]] = {}
    for opName in subOpsNames:
        nextOpsNamesList: list[str] = nextOpsDict[opName]

        if opName in modelOutputNames or len(nextOpsNamesList) == 0:
            ## This is an output layer of the model
            outputList = Utils.convertToList(allOpsDict[opName].output)
            outputOpsDict.setdefault(opName, [])
            outputOpsDict[opName].extend([x for x in range(0, len(outputList))])
        else:
            for nextOpName in nextOpsNamesList:
                if nextOpName not in subOpsNames:
                    outputOpsDict.setdefault(opName, [])
                    tensorIdxs = findPrevOperationNeededOutput(
                        allOpsDict[nextOpName], allOpsDict[opName]
                    )
                    outputOpsDict[opName].extend(tensorIdxs)
    return {opName: set(outputOpsDict[opName]) for opName in outputOpsDict.keys()}


def buildSubModel(
    subOpsNames: list[str],
    allOpsDict: dict[str, keras.Operation],
    prevOpsDict: dict[str, set[str]],
    nextOpsDict: dict[str, set[str]],
    inputLayarNames: list[str],
    outputLayerNames: list[str],
    subModIdx: int,
) -> keras.Model:

    print(f"Processing {subModIdx}")
    inputOpsDict: dict[str, set[int]] = findInputOpsList(
        subOpsNames, allOpsDict, prevOpsDict
    )
    outputOpsDict: dict[str, set[int]] = findOutputOpsList(
        subOpsNames, allOpsDict, nextOpsDict, outputLayerNames
    )

    subModel = Reconstruct.reconstructModel(
        subOpsNames,
        allOpsDict,
        prevOpsDict,
        inputOpsDict,
        outputOpsDict,
        [],  ## Assuming Unnested Model
    )

    return subModel


def modelSplit(model: keras.Model, maxLayerNum: int) -> list[keras.Model]:
    allOpsList: list[keras.Operation] = Utils.findAllOpsList(model)
    allOpsDict: dict[str, keras.Operation] = Utils.findAllOpsDict(model)

    inputLayersNames: list[str] = Utils.findInputLayers(model)
    outputLayerNames: list[str] = model.output_names

    prevOpsDict, nextOpsDict = Utils.findConnections(model)
    subModels = []
    modIdx = 0
    for i in range(0, len(allOpsList), maxLayerNum):
        subOps = allOpsList[i : min(len(allOpsList), i + maxLayerNum)]
        subOpsNames = [x.name for x in subOps]

        subModel: keras.Model = buildSubModel(
            subOpsNames,
            allOpsDict,
            prevOpsDict,
            nextOpsDict,
            inputLayersNames,
            outputLayerNames,
            modIdx,
        )

        subModels.append(subModel)
        modIdx += 1

    return subModels
