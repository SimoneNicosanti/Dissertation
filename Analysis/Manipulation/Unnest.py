import keras
from Manipulation import Reconstruct, Utils


def findSubModels(model: keras.Model) -> list[keras.Model]:
    subModels: list[keras.Model] = []
    opsQueue: list[keras.Operation] = model.operations

    while opsQueue:
        currOp: keras.Operation = opsQueue.pop()
        if isinstance(currOp, keras.Model):
            subSubModels: list[keras.Model] = findSubModels(currOp)
            subModels.extend(subSubModels)

            subModels.append(currOp)
    return subModels


def unnestModel(model: keras.Model) -> keras.Model:
    prevOpsDict: dict[str, set[str]] = Utils.findPrevConnections(model)
    allOpsDict: dict[str, keras.Operation] = Utils.findAllOpsDict(model)

    inputOpsList: list[str] = Utils.findInputLayers(model)
    inputOpsDict = {opName: set([0]) for opName in inputOpsList}

    outputOpsList: list[str] = model.output_names
    outputOpsDict = {}
    for outName in outputOpsList:
        outOp: keras.Operation = allOpsDict[outName]
        opOutputList = Utils.convertToList(outOp.output)
        outputOpsDict[outName] = set([x for x in range(0, len(opOutputList))])

    allSubModels: list[keras.Model] = findSubModels(model)

    newModel = Reconstruct.reconstructModel(
        allOpsDict.keys(),
        allOpsDict,
        prevOpsDict,
        inputOpsDict,
        outputOpsDict,
        allSubModels,
    )

    # for op in newModel.operations:
    #     op.name = f"unnested_{op.name}"

    return newModel
