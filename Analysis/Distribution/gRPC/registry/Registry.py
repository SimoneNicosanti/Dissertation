from proto import registry_pb2_grpc
from proto.registry_pb2 import (
    Empty,
    LayerInfo,
    LayerPosition,
    RegisterResponse,
    ServerInfo,
)


class Registry(registry_pb2_grpc.RegisterServicer):

    def __init__(self, partsNum: int, outputsNames: list[str]) -> None:
        super().__init__()
        self.layersPositions: dict[tuple, list[ServerInfo]] = {}
        self.nextServerIdx = 0
        self.partsNum = partsNum
        self.outputsNames: list[str] = outputsNames

    def getLayerPosition(self, request: LayerInfo, context):
        print(f"Request For {request.layerName}")
        key = (request.modelName, request.layerName)
        layerPosition = self.layersPositions[key][
            0
        ]  ## It is a list of servers --> Get the first
        return layerPosition

    def registerLayer(self, request: LayerPosition, context):
        print(f"Registred {request.layers}")
        modelName: str = request.modelName
        layers: list = request.layers
        serverInfo: ServerInfo = request.serverInfo

        for layer in layers:
            key = (modelName, layer)
            if key not in self.layersPositions:
                self.layersPositions[key] = []
            self.layersPositions[key].append(serverInfo)

        return Empty()

    def registerServer(self, request: ServerInfo, context) -> RegisterResponse:
        print(f"Sending Index >>> {self.nextServerIdx}")
        # response = RegisterResponse(subModelIdx=self.nextServerIdx)
        response = RegisterResponse(
            mainModelName="",
            subModelIdx=self.nextServerIdx,
            outputsNames=self.outputsNames,
        )
        self.nextServerIdx = (self.nextServerIdx + 1) % self.partsNum
        return response
