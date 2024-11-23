import importlib
import inspect
from concurrent import futures
from typing import Callable

import grpc
import keras
from proto import registry_pb2_grpc, server_pb2_grpc
from proto.registry_pb2 import LayerInfo, LayerPosition
from Service import Service

## Ricerca del prossimo livello della rete
## Algoritmo tipo chord basato su hash --> Ricerca distribuita senza chiedere sempre al registry:
##  * Nome del livello
##  * Posizione del livello nella rete


@keras.saving.register_keras_serializable()
class OperationWrapper(keras.Layer):

    def __init__(
        self,
        operationInfo,
        activity_regularizer=None,
        trainable=True,
        dtype=None,
        autocast=True,
        name=None,
        **kwargs,
    ) -> None:
        super().__init__(
            activity_regularizer=activity_regularizer,
            trainable=trainable,
            dtype=dtype,
            autocast=autocast,
            name=operationInfo["config"]["name"],
            **kwargs,
        )
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

    def getInput(self):
        inputTensors = [x for x in self.args if isinstance(x, keras.KerasTensor)]
        if len(inputTensors) == 1:
            input = inputTensors[0]
        else:
            input = inputTensors

        return input

    def parseArgs(self, node, argsList: list):
        if isinstance(node, dict):
            ## Keras Tensor --> Append PlaceHolder
            argsList.append(
                keras.Input(
                    shape=node["config"]["shape"][1:], dtype=node["config"]["dtype"]
                )
            )
        elif isinstance(node, list) or isinstance(node, tuple):
            ## Wrapped Input --> Go to other Level
            for elem in node:
                self.parseArgs(elem, argsList)
        else:
            ## Simple Object --> Use It
            argsList.append(node)

    def call(self, args):
        if not isinstance(args, list):
            argsList = [args]
        else:
            argsList = args

        argsDict = {}
        j = 0
        for i in range(0, len(self.args)):
            key = self.argNames[i]
            if isinstance(self.args[i], keras.KerasTensor):
                argsDict[key] = argsList[j]
                j += 1
            else:
                argsDict[key] = self.args[i]
        result = self.operation.call(**argsDict)

        return result


def main():
    model: keras.Model = keras.applications.MobileNetV3Large()
    # with open("res_net.json", "w") as f:
    #     f.write(model.to_json())

    allOps, opsInfoDict, prevOpsDict, nextOpsDict = modelParse(model)
    # print(len(allOps), len(model.layers))
    validLayersName = [layer.name for layer in model.layers]

    with grpc.insecure_channel("localhost:5000") as registryChann:
        registry: registry_pb2_grpc.RegisterStub = registry_pb2_grpc.RegisterStub(
            registryChann
        )

        portNum = 9000
        allServer = []
        MAX_LAYERS_PER_SERVICE = 30
        for i in range(0, len(allOps), MAX_LAYERS_PER_SERVICE):

            opsSubList = allOps[i : min(i + MAX_LAYERS_PER_SERVICE, len(allOps))]

            ## Subsribing Layer with a service
            for op in opsSubList:
                layerInfo: LayerInfo = LayerInfo(modelName="", layerName=op)
                layerPosition: LayerPosition = LayerPosition(
                    layerInfo=layerInfo, layerHost="localhost", layerPort=portNum
                )
                registry.registerLayer(layerPosition)

            subPrevOps = {op: prevOpsDict[op] for op in opsSubList}
            subNextOps = {op: nextOpsDict[op] for op in opsSubList}

            callables = buildCallables(opsSubList, opsInfoDict, model, validLayersName)

            server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=MAX_LAYERS_PER_SERVICE)
            )
            service = Service(callables, subPrevOps, subNextOps)
            server_pb2_grpc.add_ServerServicer_to_server(service, server)
            server.add_insecure_port(f"[::]:{portNum}")
            server.start()
            print("SERVER STARTED")
            allServer.append(server)
            portNum += 1

        for server in allServer:
            server.wait_for_termination()


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


def saveModels(callables: dict[str, Callable]):
    for opName in callables:
        if isinstance(callables[opName], keras.layers.InputLayer):
            continue

        layer: keras.Layer = callables[opName]
        if isinstance(layer, OperationWrapper):
            layerInput = layer.getInput()
        else:
            layerInput = layer.input
        layerWrapper = keras.Model(inputs=layerInput, outputs=layer(layerInput))
        keras.saving.save_model(
            layerWrapper,
            f"../../../../models/registry/{layer.name}.keras",
        )


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
