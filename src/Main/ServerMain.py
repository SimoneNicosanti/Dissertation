import os
import socket
from concurrent import futures

import grpc

from Common import ConfigReader
from proto_compiled.ping_pb2_grpc import add_PingServicer_to_server
from proto_compiled.register_pb2 import ReachabilityInfo, ServerId
from proto_compiled.register_pb2_grpc import RegisterStub
from proto_compiled.server_pb2_grpc import (
    add_AssigneeServicer_to_server,
    add_InferenceServicer_to_server,
)
from Server.Assignee.Assignee import Fetcher
from Server.Inference.IntermediateServer import IntermediateServer
from Server.Monitor.ServerMonitor import ServerMonitor
from Server.Ping.PingServer import PingServer


def main():

    ## Register to Registry
    ## Start Assignee

    dir_list = ConfigReader.ConfigReader("./config/config.ini").read_all_dirs(
        "inference_dirs"
    )
    for dir in dir_list:
        os.makedirs(dir, exist_ok=True)

    register_response: ServerId = register_to_registry()

    inferencer, inference_server = start_inference_server()
    assignee_server = start_assignee_server(register_response.server_id, inferencer)
    ping_server = start_ping_server()

    server_monitor = ServerMonitor(register_response.server_id)
    server_monitor.init_monitoring()

    assignee_server.wait_for_termination()
    inference_server.wait_for_termination()
    ping_server.wait_for_termination()
    pass


def register_to_registry():

    registry_addr = ConfigReader.ConfigReader("./config/config.ini").read_str(
        "addresses", "REGISTRY_ADDR"
    )
    registry_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
        "ports", "REGISTRY_PORT"
    )

    register_stub = RegisterStub(
        grpc.insecure_channel("{}:{}".format(registry_addr, registry_port))
    )
    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)

    assignment_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
        "ports", "ASSIGNMENT_PORT"
    )
    inference_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
        "ports", "INFERENCE_PORT"
    )
    ping_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
        "ports", "PING_PORT"
    )
    reachability_info = ReachabilityInfo(
        ip_address=ip_addr,
        assignment_port=assignment_port,
        inference_port=inference_port,
        ping_port=ping_port,
    )
    register_response = register_stub.register_server(reachability_info)

    return register_response


def start_assignee_server(server_id: int, intermediate_server):

    assignment_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
        "ports", "ASSIGNMENT_PORT"
    )

    models_dir = ConfigReader.ConfigReader("./config/config.ini").read_str(
        "inference_dirs", "MODELS_DIR"
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fetcher = Fetcher(server_id, models_dir, intermediate_server)
    add_AssigneeServicer_to_server(fetcher, server)
    server.add_insecure_port(f"[::]:{assignment_port}")
    print(f"Assignee Server running on port {assignment_port}...")

    server.start()
    return server


def start_inference_server():
    inference_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
        "ports", "INFERENCE_PORT"
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inferencer = IntermediateServer()
    add_InferenceServicer_to_server(inferencer, server)
    server.add_insecure_port(f"[::]:{inference_port}")

    server.start()
    print(f"Inference Server running on port {inference_port}...")
    return inferencer, server


def start_ping_server():

    ping_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
        "ports", "PING_PORT"
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_PingServicer_to_server(PingServer(), server)

    server.add_insecure_port(f"[::]:{ping_port}")
    server.start()
    print(f"Ping Server running on port {ping_port}...")

    return server


if __name__ == "__main__":
    # import multiprocessing as mp

    # mp.set_start_method("spawn")
    main()
