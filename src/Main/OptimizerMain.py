import time
from concurrent import futures

import grpc
from Optimizer.OptimizationServer import OptmizationServer

from proto_compiled.optimizer_pb2_grpc import add_OptimizationServicer_to_server

OPTIMIZER_PORT = 50060


def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_OptimizationServicer_to_server(OptmizationServer(), server)
    port = server.add_insecure_port(f"[::]:{OPTIMIZER_PORT}")
    print(port)
    server.start()
    print(f"gRPC Optimizer Server running on port {OPTIMIZER_PORT}...")

    server.wait_for_termination()

    pass


if __name__ == "__main__":
    main()
