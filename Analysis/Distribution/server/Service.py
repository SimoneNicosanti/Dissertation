from typing import Callable

import numpy as np
import rpyc


class RequestPool:

    def __init__(self):
        self.requestMap: dict[tuple[str, int], list] = (
            {}
        )  ## Maps (Layer Name, requestId) --> list of inputs

    def addRequest(self, layerName, requestId, input):
        key = (layerName, requestId)
        if key not in self.requestMap:
            self.requestMap[key] = []
        self.requestMap[key].append(input)

    def getRequestInfo(self, layerName, requestId):
        key = (layerName, requestId)
        return self.requestMap[key]

    def removeRequestInfo(self, layerName, requestId):
        key = (layerName, requestId)
        self.requestMap.pop(key)


class LayerService(rpyc.Service):

    requestPool = RequestPool()

    def __init__(self, ops: dict[str, Callable], prevOps: set[str], nextOps: set[str]):
        super().__init__()
        self.ops = ops
        self.prevOps = prevOps
        self.nextOps = nextOps
        self.registryConnection = rpyc.connect("localhost", 8000)

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def sendToNextLayer(self, layerName, layerResult: np.ndarray, requestId):
        convertedResult = np.array(layerResult)
        currNextLayers = self.nextOps[layerName]
        # print(f"{layerName} Sending to >>> {currNextLayers}")
        if len(currNextLayers) == 0:
            ## This is the last layer of the network
            return (
                convertedResult.tobytes(),
                convertedResult.shape,
                convertedResult.dtype.name,
            )

        for nextLayerName in currNextLayers:
            if nextLayerName in self.ops:
                processFunction: Callable = self.exposed_processLayer
            else:

                nextLayerHost: tuple = self.registryConnection.root.getLayerHost(
                    nextLayerName
                )
                nextLayerConnection = rpyc.connect(
                    nextLayerHost[0],
                    nextLayerHost[1],
                    config={"sync_request_timeout": None},
                )

                processFunction: Callable = nextLayerConnection.root.processLayer

            returnedValue = processFunction(
                nextLayerName,
                convertedResult.tobytes(),
                convertedResult.shape,
                convertedResult.dtype.name,
                requestId,
            )

            if returnedValue is not None:
                ## Next Layer has all its inputs and has returned a valid value
                return returnedValue

    def exposed_processLayer(
        self, opName: str, input: list, shape, dtype, requestId: int
    ):  # this is an exposed method
        # print(f"Processing {opName} >>> {shape}, {dtype}")
        convertedInput = np.frombuffer(input, dtype=dtype).reshape(shape)

        currOp: Callable = self.ops[opName]
        layerInputNum: int = len(self.prevOps[opName])
        if layerInputNum == 0:
            ## This is the input layer --> Forward directly to next layer
            return self.sendToNextLayer(opName, convertedInput, requestId)

        LayerService.requestPool.addRequest(opName, requestId, convertedInput)
        collectedInputs = LayerService.requestPool.getRequestInfo(opName, requestId)

        if layerInputNum == len(collectedInputs):
            ## We have collected all the inputs for the layer --> Process and send to next
            if layerInputNum == 1:
                layerResult = currOp(collectedInputs[0])
            else:
                layerResult = currOp(collectedInputs)
            LayerService.requestPool.removeRequestInfo(opName, requestId)

            return self.sendToNextLayer(opName, layerResult, requestId)

        else:
            ## We have to wait for other inputs for this layer --> We wait for them and return Null
            return None
