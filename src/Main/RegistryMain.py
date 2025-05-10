from concurrent import futures

import grpc

from Common import ConfigReader
from proto_compiled.register_pb2_grpc import add_RegisterServicer_to_server
from proto_compiled.state_pool_pb2_grpc import add_StatePoolServicer_to_server
from Registry.RegistryServer import RegistryServer
from StatePool.StatePoolServer import StatePoolServer


def main():

    config_reader = ConfigReader.ConfigReader()

    registry_addr = config_reader.read_str("addresses", "REGISTRY_ADDR")
    registry_port = config_reader.read_int("ports", "REGISTRY_PORT")
    state_pool_port = config_reader.read_int("ports", "STATE_POOL_PORT")

    registry_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_RegisterServicer_to_server(RegistryServer(), registry_server)
    registry_server.add_insecure_port("{}:{}".format(registry_addr, registry_port))
    registry_server.start()
    print("gRPC Registry Server running on port {}...".format(registry_port))

    state_pool_server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    add_StatePoolServicer_to_server(StatePoolServer(), state_pool_server)
    state_pool_server.add_insecure_port("{}:{}".format(registry_addr, state_pool_port))
    state_pool_server.start()
    print("gRPC StatePool Server running on port {}...".format(state_pool_port))

    registry_server.wait_for_termination()
    state_pool_server.wait_for_termination()


if __name__ == "__main__":
    main()
