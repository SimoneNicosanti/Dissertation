import time
from concurrent import futures

import grpc

from proto.register_pb2_grpc import add_RegisterServicer_to_server
from RegistryServer import RegistryServer


def main():

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_RegisterServicer_to_server(RegistryServer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("gRPC Server running on port 50051...")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    main()
