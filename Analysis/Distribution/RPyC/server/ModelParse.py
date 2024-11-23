import keras
import keras.src


def parse_1(model: keras.Model):
    config = model.get_config()
    layersList = config["layers"]
    layersInfoDict = {}
    for layerInfo in layersList:
        layerName = layerInfo["name"]
        layersInfoDict[layerName] = layerInfo

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
            currInfo = layersInfoDict[currOp]
            inboundNodes = currInfo["inbound_nodes"]
            histories = []
            ## For each input node find its hisyory
            for inboundNode in inboundNodes:
                findHistoryInDepth(inboundNode, histories)

            ## Updating Nodes connections for next hop
            for hist in histories:
                prevOpsDict[currOp].add(hist)

                if hist not in nextOpsDict:
                    nextOpsDict[hist] = set()
                nextOpsDict[hist].add(currOp)

            ## Updating Parsing Queue
            for hist in histories:
                if hist not in queue:
                    queue.append(hist)

    for key in nextOpsDict:
        print(f"{prevOpsDict[key]} >>> {key} >>> {nextOpsDict[key]}")


def main():
    model = keras.applications.MobileNetV3Large()
    # parse_1(model)

    layerSet = set()
    findLayers(model, layerSet)
    print(len(layerSet))


def findLayers(subMod, layerSet):
    if hasattr(subMod, "layers"):
        for layer in subMod.layers:
            findLayers(layer, layerSet)
    else:
        layerSet.add(subMod.name)


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


if __name__ == "__main__":
    main()
