from concurrent import futures

import grpc

from proto_compiled.register_pb2_grpc import add_RegisterServicer_to_server
from proto_compiled.state_pool_pb2_grpc import add_StatePoolServicer_to_server
from Registry.RegistryServer import RegistryServer
from StatePool.StatePoolServer import StatePoolServer

REGISTRY_PORT = 50051
STATE_POOL_PORT = 50052


def main():

    registry_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_RegisterServicer_to_server(RegistryServer(), registry_server)
    registry_server.add_insecure_port("[::]:{}".format(REGISTRY_PORT))
    registry_server.start()
    print("gRPC Registry Server running on port 50051...")

    state_pool_server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    add_StatePoolServicer_to_server(StatePoolServer(), state_pool_server)
    state_pool_server.add_insecure_port("[::]:{}".format(STATE_POOL_PORT))
    state_pool_server.start()
    print("gRPC StatePool Server running on port 50052...")

    registry_server.wait_for_termination()
    state_pool_server.wait_for_termination()


if __name__ == "__main__":
    main()
