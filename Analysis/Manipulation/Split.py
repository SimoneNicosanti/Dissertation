import keras
from Manipulation import Utils


def buildSubModel(
    subOpsNames: list[str],
    allOpsDict: dict[str, keras.Operation],
    prevOpsDict: dict[str, set[str]],
    nextOpsDict: dict[str, set[str]],
    subModIdx: int,
) -> keras.Model:
    subModelInput = buildSubModelInput(subOpsNames, allOpsDict, prevOpsDict)
    subModelOutput = buildSubModelOutput(subOpsNames, allOpsDict, nextOpsDict)

    # print(subOps)
    subModel = keras.Model(
        inputs=subModelInput, outputs=subModelOutput, name=f"SubMod_{subModIdx}"
    )

    return subModel


def modelSplit(model: keras.Model, maxLayerNum: int) -> list[keras.Model]:
    allOpsList: list[keras.Operation] = Utils.findAllOpsList(model)
    allOpsDict: dict[str, keras.Operation] = Utils.findAllOpsDict(model)

    prevOpsDict, nextOpsDict = Utils.findConnections(model)
    subModels = []
    modIdx = 0
    for i in range(0, len(allOpsList), maxLayerNum):
        subOps = allOpsList[i : min(len(allOpsList), i + maxLayerNum)]
        subOpsNames = [x.name for x in subOps]

        subModel: keras.Model = buildSubModel(
            subOpsNames, allOpsDict, prevOpsDict, nextOpsDict, modIdx
        )

        subModels.append(subModel)
        modIdx += 1

    return subModels


def buildSubModelInput(
    subOpsNames: list[str],
    allOpsDict: dict[str, keras.Operation],
    prevOpsDict: dict[str, set[str]],
) -> dict[str, keras.KerasTensor]:
    subModelInput = {}
    for opName in subOpsNames:
        currOp: keras.Operation = allOpsDict[opName]
        if len(prevOpsDict[opName]) == 0:
            ## This is an Input Layer
            layerOutputs = Utils.convertToList(currOp.output)
            for i, inp in enumerate(layerOutputs):
                subModelInput[f"input_{i}"] = inp
        else:
            ## This is an intermediate Op
            for prevOpName in prevOpsDict[opName]:
                if prevOpName not in subOpsNames:
                    prevOpNeededOutput = findPrevOperationNeededOutput(
                        allOpsDict[opName], allOpsDict[prevOpName]
                    )
                    for _, out in enumerate(prevOpNeededOutput):
                        # newInput = keras.Input(tensor=prevLayerOut)
                        subModelInput[out.name] = out
    return subModelInput


def findPrevOperationNeededOutput(currOp: keras.Operation, prevOp: keras.Operation):
    prevOpOutputList = Utils.convertToList(prevOp.output)
    currOpInputList = Utils.convertToList(currOp.input)

    prevNeededOutputs = []

    for currOpInput in currOpInputList:
        inputHist = currOpInput._keras_history
        inputOriginOpName, tensorIndex = (
            inputHist.operation.name,
            inputHist.tensor_index,
        )

        if inputOriginOpName == prevOp.name:
            prevNeededOutputs.append(prevOpOutputList[tensorIndex])

    return prevNeededOutputs


def buildSubModelOutput(
    subOpsNames: list[str],
    allOpsDict: dict[str, keras.Operation],
    nextOpsDict: dict[str, set[str]],
) -> dict[str, keras.KerasTensor]:
    subModelOutput = {}
    idx = 0
    for opName in subOpsNames:
        currOp: keras.Operation = allOpsDict[opName]
        layerOutput = Utils.convertToList(currOp.output)

        if len(nextOpsDict[opName]) == 0:
            ## Output Layer
            for _, out in enumerate(layerOutput):
                subModelOutput[f"output_{idx}"] = out
                idx += 1
        else:
            for nextOpName in nextOpsDict[opName]:
                if nextOpName not in subOpsNames:
                    for _, out in enumerate(layerOutput):
                        subModelOutput[out.name] = out
    return subModelOutput
