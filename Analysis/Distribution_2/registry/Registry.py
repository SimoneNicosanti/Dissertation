from proto import registry_pb2_grpc
from proto.registry_pb2 import Empty, LayerInfo, LayerPosition, Model


class Registry(registry_pb2_grpc.RegisterServicer):

    def __init__(self) -> None:
        super().__init__()
        self.layersInfo: dict[str, LayerPosition] = {}
        self.modelsInfo: dict[str, Model] = {}

    def registerModel(self, request: Model, context):
        print(f"REGISTERED MODEL {request.modelName}")
        return LayerPosition(
            layerInfo=LayerInfo(modelName="", layerName=""), layerHost="", layerPort=0
        )

    def registerLayer(self, request: LayerPosition, context):
        key = (request.layerInfo.modelName, request.layerInfo.layerName)
        print(f"Registered >>> {request.layerHost}:{request.layerPort}")
        self.layersInfo[key] = request
        return Empty()

    def getLayerPosition(self, request: LayerInfo, context):
        layerPosition = self.layersInfo[(request.modelName, request.layerName)]
        print(
            f"Asked for Layer {request.layerName} at {layerPosition.layerHost}:{layerPosition.layerPort}"
        )
        return layerPosition
