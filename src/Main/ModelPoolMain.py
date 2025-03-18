import os
from concurrent import futures

import grpc

from Common import ConfigReader
from ModelPool.PoolServer import PoolServer
from proto_compiled.pool_pb2_grpc import add_ModelPoolServicer_to_server


def main():

    dir_list = ConfigReader.ConfigReader("./config/config.ini").read_all_dirs(
        "model_pool_dirs"
    )
    for dir in dir_list:
        os.makedirs(dir, exist_ok=True)

    model_pool_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
        "ports", "MODEL_POOL_PORT"
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pool_server = PoolServer()
    add_ModelPoolServicer_to_server(pool_server, server)
    server.add_insecure_port(f"[::]:{model_pool_port}")
    print("Starting Model Pool Server...")
    server.start()
    print(f"gRPC Model Pool Server running on port {model_pool_port}...")

    server.wait_for_termination()

    pass


if __name__ == "__main__":
    main()
