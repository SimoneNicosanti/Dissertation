from ModelManager import ModelManager
from proto import registry_pb2_grpc
from proto.registry_pb2 import LayerInfo, LayerList, RegisterResponse, ServerInfo


class Registry(registry_pb2_grpc.RegisterServicer):

    def __init__(self, modelManager: ModelManager) -> None:
        super().__init__()
        self.layersPositions: dict[str, list[ServerInfo]] = {}
        self.modelManager = modelManager
        self.nextServerIdx = 0

    def getLayerPosition(self, request: LayerInfo, context):
        key = (request.modelName, request.layerName)
        layerPosition = self.layersPositions[key][
            0
        ]  ## It is a list of servers --> Get the first
        return layerPosition

    def registerServer(self, request: ServerInfo, context) -> RegisterResponse:
        opsSubList, subPrevOps, subNextOps = self.modelManager.getSubModelInfo(
            self.nextServerIdx
        )

        ## Appending to layer server
        for op in opsSubList:
            if op not in self.layersPositions:
                self.layersPositions[op] = []
            self.layersPositions[op].append(request)

        layerList: LayerList = LayerList(layers=opsSubList)
        prevLayers = {op: LayerList(layers=subPrevOps[op]) for op in opsSubList}
        nextLayers = {op: LayerList(layers=subNextOps[op]) for op in opsSubList}
        response = RegisterResponse(
            layers=layerList, prevLayers=prevLayers, nextLayers=nextLayers
        )

        self.nextServerIdx += 1
        return response
