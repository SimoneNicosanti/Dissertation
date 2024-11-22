import importlib
import inspect

import keras


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
    model = keras.applications.MobileNetV3Large()
    # parse_1(model)
    allOps, opsInfoDict, prevOpsDict, nextOpsDict = modelParse(model)
    buildCallables(allOps, opsInfoDict, model, [layer.name for layer in model.layers])

    model_1: keras.Model = keras.models.load_model(
        "../../../models/registry/activation_7.keras"
    )
    print(model_1.input)


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

    for opName in callables:
        if isinstance(callables[opName], keras.Layer):
            print(opName)
            layer: keras.Layer = callables[opName]
            layerWrapper = keras.Model(inputs=layer.input, outputs=layer(layer.input))
            keras.saving.save_model(
                layerWrapper,
                f"../../../models/registry/{callables[opName].name}.keras",
            )

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
