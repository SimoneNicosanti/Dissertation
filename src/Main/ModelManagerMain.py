import os
from concurrent import futures

import grpc

from Common import ConfigReader
from ModelDivider.ModelDivideServer import ModelDivideServer
from ModelPool.PoolServer import PoolServer
from ModelProfiler.ModelProfileServer import ModelProfileServer
from proto_compiled.model_divide_pb2_grpc import add_ModelDivideServicer_to_server
from proto_compiled.model_pool_pb2_grpc import add_ModelPoolServicer_to_server
from proto_compiled.model_profile_pb2_grpc import add_ModelProfileServicer_to_server


def main():

    dir_list = ConfigReader.ConfigReader().read_all_dirs("model_pool_dirs")
    for dir in dir_list:
        os.makedirs(dir, exist_ok=True)

    dir_list = ConfigReader.ConfigReader().read_all_dirs("model_profiler_dirs")
    for dir in dir_list:
        os.makedirs(dir, exist_ok=True)

    model_pool_server: grpc.Server = start_model_pool()
    model_divider_server: grpc.Server = start_model_divider()
    model_profile_server: grpc.Server = start_model_profiler()

    model_pool_server.wait_for_termination()
    model_divider_server.wait_for_termination()
    model_profile_server.wait_for_termination()

    pass


def start_model_pool() -> grpc.Server:
    model_pool_port = ConfigReader.ConfigReader().read_int("ports", "MODEL_POOL_PORT")
    model_pool_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    pool_servicer = PoolServer()
    add_ModelPoolServicer_to_server(pool_servicer, model_pool_server)

    model_pool_server.add_insecure_port(f"[::]:{model_pool_port}")
    model_pool_server.start()

    print(f"gRPC Model Pool Server running on port {model_pool_port}...")

    return model_pool_server


def start_model_divider():
    model_divider_port = ConfigReader.ConfigReader().read_int(
        "ports", "MODEL_DIVIDER_PORT"
    )
    model_divider_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    divider_servicer = ModelDivideServer()
    add_ModelDivideServicer_to_server(divider_servicer, model_divider_server)

    model_divider_server.add_insecure_port(f"[::]:{model_divider_port}")
    model_divider_server.start()

    print(f"gRPC Model Divider Server running on port {model_divider_port}...")

    return model_divider_server


def start_model_profiler():
    model_profile_port = ConfigReader.ConfigReader().read_int(
        "ports", "MODEL_PROFILER_PORT"
    )
    model_profile_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    profile_servicer = ModelProfileServer()
    add_ModelProfileServicer_to_server(profile_servicer, model_profile_server)

    model_profile_server.add_insecure_port(f"[::]:{model_profile_port}")
    model_profile_server.start()

    print(f"gRPC Model Profiler Server running on port {model_profile_port}...")

    return model_profile_server
    pass


if __name__ == "__main__":
    main()
