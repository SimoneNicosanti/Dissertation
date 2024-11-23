import socket
from concurrent import futures

import grpc
import keras
import psutil
from proto import registry_pb2_grpc, server_pb2_grpc
from proto.registry_pb2 import RegisterResponse, ServerInfo
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

    with grpc.insecure_channel("registry:5000") as registryChann:
        registry: registry_pb2_grpc.RegisterStub = registry_pb2_grpc.RegisterStub(
            registryChann
        )

        portNum = 9000

        serverInfo: ServerInfo = ServerInfo(hostName=addresses[0], portNum=portNum)
        registerResponse: RegisterResponse = registry.registerServer(serverInfo)

        callables = loadLayers(registerResponse.layers.layers)
        prevOps = {
            op: registerResponse.prevLayers[op].layers
            for op in registerResponse.prevLayers
        }
        nextOps = {
            op: registerResponse.nextLayers[op].layers
            for op in registerResponse.nextLayers
        }

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=30))
        service = Service(callables, prevOps, nextOps)
        server_pb2_grpc.add_ServerServicer_to_server(service, server)
        server.add_insecure_port(f"[::]:{portNum}")
        server.start()
        print("SERVER STARTED")
        server.wait_for_termination()


def loadLayers(layerList: list[str]):
    callables = []
    for layerName in layerList:
        layer = keras.saving.load_model(f"/callables/{layerName}.keras")
        callables.append(layer)
    return callables


if __name__ == "__main__":
    main()
