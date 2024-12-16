import socket
from concurrent import futures

import grpc
import keras
import psutil
from ai_edge_litert.interpreter import Interpreter, SignatureRunner
from proto import registry_pb2_grpc, server_pb2_grpc
from proto.registry_pb2 import LayerPosition, RegisterResponse, ServerInfo
from ServiceKeras import ServiceKeras
from ServiceLite import ServiceLite


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
    print(addresses)

    with grpc.insecure_channel("registry:5000") as registryChann:
        registry: registry_pb2_grpc.RegisterStub = registry_pb2_grpc.RegisterStub(
            registryChann
        )

        portNum = 9000

        serverInfo: ServerInfo = ServerInfo(hostName=addresses[0], portNum=portNum)
        resp: RegisterResponse = registry.registerServer(serverInfo)

        print("Index ", resp.subModelIdx)
        print(f"Main Model Name >> {resp.mainModelName}")
        print(f"Output Names >> {resp.outputsNames}")

        # service = buildServiceKeras(resp, serverInfo, registry)
        service = buildServiceLite(resp, serverInfo, registry)

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=30))

        server_pb2_grpc.add_ServerServicer_to_server(service, server)
        server.add_insecure_port(f"[::]:{portNum}")
        server.start()
        print("SERVER STARTED")
        server.wait_for_termination()


def buildServiceKeras(resp, serverInfo, registry):
    subModel: keras.Model = keras.saving.load_model(
        f"/models/SubModel_{resp.subModelIdx}.keras"
    )
    layers = [inputLayer for inputLayer in subModel.input]
    layersPosition = LayerPosition(modelName="", layers=layers, serverInfo=serverInfo)
    registry.registerLayer(layersPosition)

    return ServiceKeras(subModel, resp.outputsNames)


def buildServiceLite(resp, serverInfo, registry):
    interpreter: Interpreter = Interpreter(
        model_path=f"/models/SubModel_{resp.subModelIdx}.tflite"
    )
    signatureRunner: SignatureRunner = interpreter.get_signature_runner(
        "serving_default"
    )
    layers = list(signatureRunner.get_input_details().keys())
    layersPosition = LayerPosition(modelName="", layers=layers, serverInfo=serverInfo)
    registry.registerLayer(layersPosition)

    return ServiceLite(interpreter, resp.outputsNames)


if __name__ == "__main__":
    main()
