import grpc
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter, SignatureRunner
from proto import registry_pb2_grpc, server_pb2_grpc
from proto.registry_pb2 import LayerInfo, ServerInfo
from proto.server_pb2 import ModelInput, ModelOutput


class RequestPool:

    def __init__(self):
        self.requestMap: dict[int, dict[str]] = (
            {}
        )  ## Maps (Layer Name, requestId) --> list of inputs

    def addRequest(self, layerName, requestId, input):
        key = requestId
        if key not in self.requestMap:
            self.requestMap[key] = {}
        self.requestMap[key][layerName] = input

    def getRequestInfo(self, requestId):
        key = requestId
        return self.requestMap[key]

    def removeRequestInfo(self, requestId):
        key = requestId
        self.requestMap.pop(key)


class ServiceLite(server_pb2_grpc.ServerServicer):
    def __init__(self, interpreter: Interpreter, outputNames: list[str]) -> None:
        super().__init__()
        self.interpeter = interpreter
        self.model: SignatureRunner = interpreter.get_signature_runner(
            "serving_default"
        )
        self.outputNames: list[str] = outputNames
        self.requestPool = RequestPool()

        channel = grpc.insecure_channel("registry:5000")
        self.registry: registry_pb2_grpc.RegisterStub = registry_pb2_grpc.RegisterStub(
            channel
        )

    def sendToNextLayer(self, modelOutput: dict, requestId: int) -> ModelOutput:
        print("Sending to Next")
        for outName in modelOutput.keys():
            if outName in self.outputNames:
                return ModelOutput(
                    hasValue=True,
                    result={
                        key: tf.make_tensor_proto(modelOutput[key])
                        for key in modelOutput
                    },
                )
            else:
                subResult = modelOutput[outName]
                nextInput = ModelInput(
                    requestId=requestId,
                    modelName="",
                    layerName=outName,
                    tensor=tf.make_tensor_proto(subResult),
                )
                print("Converted Output")

                nextLayerHost: ServerInfo = self.registry.getLayerPosition(
                    LayerInfo(modelName="", layerName=outName)
                )
                nextHostChann = grpc.insecure_channel(
                    f"{nextLayerHost.hostName}:{nextLayerHost.portNum}"
                )
                serverStub: server_pb2_grpc.ServerStub = server_pb2_grpc.ServerStub(
                    nextHostChann
                )
                print("Sent Value")
                returnedValue: ModelOutput = serverStub.serveModel(nextInput)

        return returnedValue

    def _serveModel(self, request: ModelInput) -> ModelOutput:
        inputLayer: str = request.layerName
        requestId: int = request.requestId
        convertedInput: np.ndarray = tf.make_ndarray(request.tensor)

        self.requestPool.addRequest(inputLayer, requestId, convertedInput)
        collectedInputs = self.requestPool.getRequestInfo(requestId)
        print("Received Input")

        if len(self.model.get_input_details()) == len(collectedInputs):
            ## We have collected all the inputs for the sub model --> We can process the sub model
            print("Collected Input")
            model = self.interpeter.get_signature_runner("serving_default")
            modelOutput = model(**collectedInputs)
            print("Run Model")
            self.requestPool.removeRequestInfo(requestId)

            return self.sendToNextLayer(modelOutput, requestId)

        else:
            ## We have to wait for other inputs for this sub model --> We wait for them and return empty response
            return ModelOutput(hasValue=False, result={})

    def serveModel(self, request: ModelInput, context) -> ModelOutput:
        return self._serveModel(request)