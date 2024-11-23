from typing import Callable

import keras
import keras.src
import keras.src.ops.numpy


def modelParse(model: keras.Model):
    config = model.get_config()
    layersList = config["layers"]
    opsInfoDict = {}
    for layerInfo in layersList:
        layerName = layerInfo["name"]
        opsInfoDict[layerName] = layerInfo

    allOps = []
    queue = [x for x in model.output_names]
    nextOpsDict = {}
    prevOpsDict = {}

    while queue:
        currOp = queue.pop(0)

        if currOp not in prevOpsDict:
            prevOpsDict[currOp] = set()
        if currOp not in nextOpsDict:
            nextOpsDict[currOp] = set()

        if currOp not in allOps:
            allOps.append(currOp)
            ## Find prev ops
            currInfo = opsInfoDict[currOp]
            inboundNodes = currInfo["inbound_nodes"]
            histories = []
            ## For each input node find its hisyory
            for inboundNode in inboundNodes:
                findHistoryInDepth(inboundNode, histories)

            ## Updating Nodes connections for next hop
            ## This is actually a layer, so can add it to connections
            if currInfo["module"] == "keras.layers":
                for hist in histories:
                    prevOpsDict[currOp].add(hist)

                    if hist not in nextOpsDict:
                        nextOpsDict[hist] = set()
                    nextOpsDict[hist].add(currOp)

            ## Updating Parsing Queue
            for hist in histories:
                if hist not in queue:
                    queue.append(hist)

    # for key in allOps:
    #     print(f"{prevOpsDict[key]} >>> {key} >>> {nextOpsDict[key]}")

    # print(len(allOps), len(prevOpsDict), len(nextOpsDict))

    return allOps, opsInfoDict, prevOpsDict, nextOpsDict


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


def saveCallables(callables: dict[str, Callable]):
    for opName in callables:

        layer: keras.Layer = callables[opName]
        if isinstance(layer, keras.layers.InputLayer):
            # layerInput = keras.Input(shape=layer.output)
            continue  ## Handle this case
        elif isinstance(layer, OperationWrapper):
            layerInput = layer.getInput()
        else:
            layerInput = layer.input
        layerWrapper = keras.Model(inputs=layerInput, outputs=layer(layerInput))
        keras.saving.save_model(
            layerWrapper,
            f"/callables/{layer.name}.keras",
        )


def buildCallables(
    opsList: list[str], opsInfoDict: dict, model: keras.Model, validLayersName: str
):
    callables = {}
    for op in opsList:
        if op in validLayersName:
            callables[op] = model.get_layer(op)
        else:
            ## It is other type of operation
            callables[op] = OperationWrapper(opsInfoDict[op])

    return callables
