import socket
from concurrent import futures

import grpc
import keras
import psutil
from proto import registry_pb2_grpc, server_pb2_grpc
from proto.registry_pb2 import LayerPosition, RegisterResponse, ServerInfo
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
        resp: RegisterResponse = registry.registerServer(serverInfo)

        print("Index ", resp.subModelIdx)
        subModel: keras.Model = keras.saving.load_model(
            f"/models/SubModel_{resp.subModelIdx}.keras"
        )
        layers = [inputLayer for inputLayer in subModel.input]
        print("Input Layers >>> ", layers)
        layersPosition = LayerPosition(
            modelName="", layers=layers, serverInfo=serverInfo
        )
        registry.registerLayer(layersPosition)

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=30))
        service = Service(subModel)
        server_pb2_grpc.add_ServerServicer_to_server(service, server)
        server.add_insecure_port(f"[::]:{portNum}")
        server.start()
        print("SERVER STARTED")
        server.wait_for_termination()


if __name__ == "__main__":
    main()
