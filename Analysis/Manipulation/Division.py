import pickle

import keras
import Utils


def modelParse(model: keras.Model, maxLayerNum: int) -> list[keras.Model]:
    allOpsList: list[keras.Operation] = Utils.findAllOpsList(model)
    allOpsDict: dict[str, keras.Operation] = Utils.findAllOpsDict(model)

    prevOpsDict, nextOpsDict = Utils.findConnections(model)
    subModels = []
    modIdx = 0
    for i in range(0, len(allOpsList), maxLayerNum):
        subOps = allOpsList[i : min(len(allOpsList), i + maxLayerNum)]
        subOpsNames = [x.name for x in subOps]

        subModelInput = buildSubModelInput(subOpsNames, allOpsDict, prevOpsDict)
        subModelOutput = buildSubModelOutput(subOpsNames, allOpsDict, nextOpsDict)

        print(subOps)
        subModel = keras.Model(
            inputs=subModelInput, outputs=subModelOutput, name=f"SubMod_{modIdx}"
        )
        subModels.append(subModel)
        print(subModel.name)
        modIdx += 1

    return subModels


def buildSubModelInput(
    subOpsNames: list[str],
    allOpsDict: dict[str, keras.Operation],
    prevOpsDict: dict[str, set[str]],
):
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
):
    subModelOutput = {}
    for opName in subOpsNames:
        currOp: keras.Operation = allOpsDict[opName]
        layerOutput = Utils.convertToList(currOp.output)

        if len(nextOpsDict[opName]) == 0:
            ## Output Layer
            for i, out in enumerate(layerOutput):
                subModelOutput[f"output_{i}"] = out
        else:
            for nextOpName in nextOpsDict[opName]:
                if nextOpName not in subOpsNames:
                    for _, out in enumerate(layerOutput):
                        subModelOutput[out.name] = out
    return subModelOutput
