import os
from concurrent import futures

import grpc

from Common import ConfigReader
from Optimizer.OptimizationServer import OptmizationServer
from proto_compiled.optimizer_pb2_grpc import add_OptimizationServicer_to_server


def main():

    optimizer_port = ConfigReader.ConfigReader().read_int("ports", "OPTIMIZER_PORT")

    dir_list = ConfigReader.ConfigReader().read_all_dirs("optimizer_dirs")
    for dir in dir_list:
        os.makedirs(dir, exist_ok=True)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 1000 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1000 * 1024 * 1024),
        ],
    )
    add_OptimizationServicer_to_server(OptmizationServer(), server)
    port = server.add_insecure_port(f"[::]:{optimizer_port}")
    print(port)
    server.start()
    print(f"gRPC Optimizer Server running on port {optimizer_port}...")

    server.wait_for_termination()

    pass


if __name__ == "__main__":
    main()
