import socket
from concurrent import futures

import grpc
import keras
import psutil
from ModelParse import buildCallables, modelParse
from proto import registry_pb2_grpc, server_pb2_grpc
from proto.registry_pb2 import LayerInfo, LayerPosition
from Service import Service


## Ricerca del prossimo livello della rete
## Algoritmo tipo chord basato su hash --> Ricerca distribuita senza chiedere sempre al registry:
##  * Nome del livello
##  * Posizione del livello nella rete
def get_all_ip_addresses():
    addresses = []
    for interface, addrs in psutil.net_if_addrs().items():
        if interface != "lo":
            for addr in addrs:
                if addr.family == socket.AF_INET:  # Only IPv4
                    addresses.append(addr.address)
    return addresses


def main():
    addresses = get_all_ip_addresses()
    model: keras.Model = keras.applications.MobileNetV3Large()
    # with open("res_net.json", "w") as f:
    #     f.write(model.to_json())

    allOps, opsInfoDict, prevOpsDict, nextOpsDict = modelParse(model)
    # print(len(allOps), len(model.layers))
    validLayersName = [layer.name for layer in model.layers]

    with grpc.insecure_channel("registry:5000") as registryChann:
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
                    layerInfo=layerInfo, layerHost=addresses[0], layerPort=portNum
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


if __name__ == "__main__":
    main()
