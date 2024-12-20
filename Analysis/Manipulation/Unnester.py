import keras
from Manipulation.ModelGraph import ModelGraph
from Manipulation.OperationWrapper import OperationWrapper
from Manipulation import Utils
from Manipulation import Reconstructor


def unnestModel(model : keras.Model) :
    modelGraph : ModelGraph = ModelGraph(model)

    prevOpsDict: dict[str, set[str]] = modelGraph.prevOpsDict
    allOpsDict: dict[str, OperationWrapper] = modelGraph.allOpsDict

    inputOpsList: list[str] = modelGraph.inputOpsList
    inputOpsDict = {opName: set([0]) for opName in inputOpsList}

    outputOpsList: list[str] = modelGraph.outputOpsList
    outputOpsDict = {}
    for outName in outputOpsList:
        outOp: OperationWrapper = allOpsDict[outName]
        opOutputList = Utils.convertToList(outOp.getOpOutput())
        outputOpsDict[outName] = set([x for x in range(0, len(opOutputList))])

    newModel = Reconstructor.reconstructModel(
        allOpsDict.keys(),
        allOpsDict,
        prevOpsDict,
        inputOpsDict,
        outputOpsDict,
    )

    return newModel
