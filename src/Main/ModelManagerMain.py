import os
from concurrent import futures

import grpc

from Common import ConfigReader
from ModelManager.ModelManagerServer import ModelManagerServer
from ModelPool.PoolServer import PoolServer
from proto_compiled.model_manager_pb2_grpc import add_ModelManagerServicer_to_server
from proto_compiled.pool_pb2_grpc import add_ModelPoolServicer_to_server


def main():

    dir_list = ConfigReader.ConfigReader("./config/config.ini").read_all_dirs(
        "model_pool_dirs"
    )
    for dir in dir_list:
        os.makedirs(dir, exist_ok=True)
    
    dir_list = ConfigReader.ConfigReader("./config/config.ini").read_all_dirs(
        "model_manager_dirs"
    )
    for dir in dir_list:
        os.makedirs(dir, exist_ok=True)

    model_pool_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
        "ports", "MODEL_POOL_PORT"
    )
    model_pool_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pool_servicer = PoolServer()
    add_ModelPoolServicer_to_server(pool_servicer, model_pool_server)
    model_pool_server.add_insecure_port(f"[::]:{model_pool_port}")
    model_pool_server.start()
    print(f"gRPC Model Pool Server running on port {model_pool_port}...")

    model_manager_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
        "ports", "MODEL_MANAGER_PORT"
    )
    model_manager_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_manager_servicer = ModelManagerServer()
    add_ModelManagerServicer_to_server(model_manager_servicer, model_manager_server)
    model_manager_server.add_insecure_port(f"[::]:{model_manager_port}")
    model_manager_server.start()
    print(f"gRPC Model Manager Server running on port {model_manager_port}...")

    model_pool_server.wait_for_termination()
    model_manager_server.wait_for_termination()

    pass


if __name__ == "__main__":
    main()
