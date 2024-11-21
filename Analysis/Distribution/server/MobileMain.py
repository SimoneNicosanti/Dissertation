import importlib
import inspect
from threading import Thread

import keras
import rpyc
from rpyc.utils.server import ThreadedServer
from Service import LayerService

## Ricerca del prossimo livello della rete
## Algoritmo tipo chord basato su hash --> Ricerca distribuita senza chiedere sempre al registry:
##  * Nome del livello
##  * Posizione del livello nella rete


class OperationWrapper:

    def __init__(self, operationInfo) -> None:
        self.args = []
        for inNode in operationInfo["inbound_nodes"]:
            if "args" in inNode:
                self.parseArgs(inNode["args"], self.args)

        moduleName = operationInfo["module"]
        className = operationInfo["class_name"]
        self.operation = getattr(importlib.import_module(moduleName), className)()

        # Inspect the signature and get parameters names
        signature = inspect.signature(self.operation.call)
        self.argNames = [param.name for param in signature.parameters.values()]

    def parseArgs(self, node, argsList: list):
        if isinstance(node, dict):
            ## Keras Tensor --> Append PlaceHolder
            argsList.append(None)
        elif isinstance(node, list) or isinstance(node, tuple):
            ## Wrapped Input --> Go to other Level
            for elem in node:
                self.parseArgs(elem, argsList)
        else:
            ## Simple Object --> Use It
            argsList.append(node)

    def __call__(self, args):
        if not isinstance(args, list):
            argsList = [args]
        else:
            argsList = args

        argsDict = {}
        j = 0
        for i in range(0, len(self.args)):
            key = self.argNames[i]
            if self.args[i] is None:
                argsDict[key] = argsList[j]
                j += 1
            else:
                argsDict[key] = self.args[i]
        result = self.operation.call(**argsDict)

        return result


def main():
    model: keras.Model = keras.applications.MobileNetV3Small()
    with open("res_net.json", "w") as f:
        f.write(model.to_json())

    allOps, opsInfoDict, prevOpsDict, nextOpsDict = modelParse(model)
    print(len(allOps), len(model.layers))
    validLayersName = [layer.name for layer in model.layers]

    c = rpyc.connect("localhost", 8000)

    portNum = 9000
    allThreads = []
    MAX_LAYERS_PER_SERVICE = 30
    for i in range(0, len(allOps), MAX_LAYERS_PER_SERVICE):

        opsSubList = allOps[i : min(i + MAX_LAYERS_PER_SERVICE, len(allOps))]

        ## Subsribing Layer with a service
        for op in opsSubList:
            c.root.subscribeLayer(op, "localhost", portNum)

        subPrevOps = {op: prevOpsDict[op] for op in opsSubList}
        subNextOps = {op: nextOpsDict[op] for op in opsSubList}

        callables = buildCallables(opsSubList, opsInfoDict, model, validLayersName)

        serverThread = Thread(
            target=startServer,
            args=(callables, subPrevOps, subNextOps, portNum),
            daemon=True,
        )
        serverThread.start()
        allThreads.append(serverThread)
        portNum += 1

    for thread in allThreads:
        thread.join()


def startServer(callables, prevOpsDict, nextOpsDict, portNum):
    t = ThreadedServer(
        LayerService(callables, prevOpsDict, nextOpsDict),
        port=portNum,
        listener_timeout=None,
    )
    print("Starting Service")
    t.start()


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


if __name__ == "__main__":
    main()
