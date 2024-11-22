from typing import Callable

import grpc
import numpy as np
import tensorflow as tf
from proto import registry_pb2_grpc, server_pb2_grpc
from proto.registry_pb2 import LayerInfo, LayerPosition
from proto.server_pb2 import LayerRequest, LayerResponse


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


class Service(server_pb2_grpc.ServerServicer):
    def __init__(
        self,
        ops: dict[str, Callable],
        prevOps: dict[str, set[str]],
        nextOps: dict[str, set[str]],
    ) -> None:
        super().__init__()
        self.ops: dict[str, Callable] = ops
        self.prevOps: dict[str, set[str]] = prevOps
        self.nextOps: dict[str, set[str]] = nextOps
        self.requestPool = RequestPool()

        channel = grpc.insecure_channel("localhost:5000")
        self.registry: registry_pb2_grpc.RegisterStub = registry_pb2_grpc.RegisterStub(
            channel
        )

    def sendToNextLayer(self, opName: str, layerResult, requestId) -> LayerResponse:
        convertedResult = tf.make_tensor_proto(layerResult)
        currNextLayers = self.nextOps[opName]
        # print(f"{layerName} Sending to >>> {currNextLayers}")
        if len(currNextLayers) == 0:
            ## This is the last layer of the network
            return LayerResponse(hasValue=True, result=convertedResult)

        for nextLayerName in currNextLayers:
            if nextLayerName in self.ops:
                # print(f"Layer {nextLayerName} is in Local")
                processFunction: Callable = self._serveLayer
            else:
                # print(f"Layer {nextLayerName} is in Remote")
                nextLayerHost: LayerPosition = self.registry.getLayerPosition(
                    LayerInfo(modelName="", layerName=nextLayerName)
                )

                nextHostChann = grpc.insecure_channel(
                    f"{nextLayerHost.layerHost}:{nextLayerHost.layerPort}"
                )
                processFunction: Callable = server_pb2_grpc.ServerStub(
                    nextHostChann
                ).serveLayer

            nextLayerRequest: LayerRequest = LayerRequest(
                modelName="",
                layerName=nextLayerName,
                requestId=requestId,
                tensor=convertedResult,
            )
            returnedValue: LayerResponse = processFunction(nextLayerRequest)

            # if returnedValue.hasValue:
            ## Next Layer has all its inputs and has returned a valid value
        return returnedValue

    def _serveLayer(self, request: LayerRequest) -> LayerResponse:
        opName: str = request.layerName
        requestId: int = request.requestId
        convertedInput: tf.Tensor = tf.convert_to_tensor(
            tf.make_ndarray(request.tensor)
        )
        # print(f"{opName} Processing >>> {convertedInput.shape} {convertedInput.dtype}")

        # convertedInput = np.frombuffer(input, dtype=dtype).reshape(shape)

        currOp: Callable = self.ops[opName]
        layerInputNum: int = len(self.prevOps[opName])
        if layerInputNum == 0:
            ## This is the input layer --> Forward directly to next layer
            return self.sendToNextLayer(opName, convertedInput, requestId)

        self.requestPool.addRequest(opName, requestId, convertedInput)
        collectedInputs = self.requestPool.getRequestInfo(opName, requestId)

        if layerInputNum == len(collectedInputs):
            ## We have collected all the inputs for the layer --> Process and send to next
            if layerInputNum == 1:
                layerResult = currOp(collectedInputs[0])
            else:
                layerResult = currOp(collectedInputs)
            self.requestPool.removeRequestInfo(opName, requestId)

            return self.sendToNextLayer(opName, layerResult, requestId)

        else:
            ## We have to wait for other inputs for this layer --> We wait for them and return empty response
            return LayerResponse(
                hasValue=False, result=tf.make_tensor_proto(np.zeros(shape=(1)))
            )

    def serveLayer(self, request: LayerRequest, context) -> LayerResponse:
        return self._serveLayer(request)
