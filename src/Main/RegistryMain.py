from concurrent import futures

import grpc

from Common import ConfigReader
from proto_compiled.register_pb2_grpc import add_RegisterServicer_to_server
from Registry.RegistryServer import RegistryServer


def main():

    config_reader = ConfigReader.ConfigReader()

    registry_addr = config_reader.read_str("addresses", "REGISTRY_ADDR")
    registry_port = config_reader.read_int("ports", "REGISTRY_PORT")

    registry_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_RegisterServicer_to_server(RegistryServer(), registry_server)
    registry_server.add_insecure_port("{}:{}".format(registry_addr, registry_port))
    registry_server.start()
    print("gRPC Registry Server running on port {}...".format(registry_port))

    registry_server.wait_for_termination()


if __name__ == "__main__":
    main()
