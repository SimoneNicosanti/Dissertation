import time
from concurrent import futures

import grpc
from Registry.RegistryServer import RegistryServer

from proto_compiled.register_pb2_grpc import add_RegisterServicer_to_server


def main():

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_RegisterServicer_to_server(RegistryServer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("gRPC Server running on port 50051...")

    server.wait_for_termination()


if __name__ == "__main__":
    main()
