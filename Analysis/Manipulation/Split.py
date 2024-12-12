import keras
from Manipulation import Utils


def buildSubModel(
    subOpsNames: list[str],
    allOpsDict: dict[str, keras.Operation],
    prevOpsDict: dict[str, set[str]],
    nextOpsDict: dict[str, set[str]],
    inputLayarNames: list[str],
    outputLayerNames: list[str],
    subModIdx: int,
) -> keras.Model:
    subModelInput = buildSubModelInput(
        subOpsNames, inputLayarNames, allOpsDict, prevOpsDict
    )
    subModelOutput = buildSubModelOutput(
        subOpsNames, outputLayerNames, allOpsDict, nextOpsDict
    )

    subModel = keras.Model(
        inputs=subModelInput, outputs=subModelOutput, name=f"SubMod_{subModIdx}"
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


def buildSubModelInput(
    subOpsNames: list[str],
    inputLayersNames: list[str],
    allOpsDict: dict[str, keras.Operation],
    prevOpsDict: dict[str, set[str]],
) -> dict[str, keras.KerasTensor]:
    subModelInput = {}

    for opName in subOpsNames:
        currOp: keras.Operation = allOpsDict[opName]

        if opName in inputLayersNames:
            ## Is an input layer
            inputIdx: int = inputLayersNames.index(opName)
            subModelInput[f"input_{inputIdx}"] = currOp.output

        else:
            prevOpsNames: set[str] = prevOpsDict[opName]
            ## Other Operation
            ## Needs Output from other layer
            for prevOpName in prevOpsNames:
                prevOp: keras.Operation = allOpsDict[prevOpName]
                if prevOpName not in subOpsNames:
                    ## Needs Input from other sub model
                    neededInputs: list[keras.KerasTensor] = (
                        findPrevOperationNeededOutput(currOp, prevOp)
                    )

                    for inpTensor in neededInputs:
                        # newInput._keras_history = tensor._keras_history
                        subModelInput[inpTensor.name] = inpTensor
    return subModelInput


def findPrevOperationNeededOutput(
    currOp: keras.Operation, prevOp: keras.Operation
) -> list[keras.KerasTensor]:
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
            tensor: keras.KerasTensor = prevOpOutputList[tensorIndex]
            prevNeededOutputs.append(tensor)

    return prevNeededOutputs


def buildSubModelOutput(
    subOpsNames: list[str],
    outputLayersNames: list[str],
    allOpsDict: dict[str, keras.Operation],
    nextOpsDict: dict[str, set[str]],
) -> dict[str, keras.KerasTensor]:
    subModelOutput = {}

    for opName in subOpsNames:
        currOp: keras.Operation = allOpsDict[opName]

        if opName in outputLayersNames:
            ## Is Model Output Layer --> Set its output as output model
            ## TODO This may give problems if the operation has more than one output ??
            outIdx = outputLayersNames.index(opName)
            subModelOutput[f"output_{outIdx}"] = currOp.output

        nextOpsNames: set[str] = nextOpsDict[opName]
        for nextOpName in nextOpsNames:
            nextOp: keras.Operation = allOpsDict[nextOpName]
            if nextOpName not in subOpsNames:
                producedOutputs: list[tuple[keras.KerasTensor, int]] = (
                    findPrevOperationNeededOutput(nextOp, currOp)
                )
                for outTensor in producedOutputs:
                    subModelOutput[outTensor.name] = outTensor

    return subModelOutput
