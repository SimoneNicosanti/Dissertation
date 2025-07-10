import os
import socket
import subprocess
import threading
from concurrent import futures

import grpc

from Common import ConfigReader
from proto_compiled.ping_pb2_grpc import add_PingServicer_to_server
from proto_compiled.register_pb2 import ReachabilityInfo, ServerId
from proto_compiled.register_pb2_grpc import RegisterStub
from proto_compiled.server_pb2_grpc import (
    add_ExecutionProfileServicer_to_server,
    add_InferenceServicer_to_server,
)
from Server.FrontEnd.FrontEndServer import FrontEndServer
from Server.Inference.IntermediateServer import IntermediateServer
from Server.Monitor.ServerMonitor import ServerMonitor
from Server.Ping.PingServer import PingServer
from Server.Profiler.ExecutionProfileServer import ExecutionProfileServer


def main():

    ## Register to Registry
    ## Start Assignee

    # threading.Thread(target=start_iperf3_server).start()

    dir_list = ConfigReader.ConfigReader().read_all_dirs("server_dirs")
    for dir in dir_list:
        os.makedirs(dir, exist_ok=True)

    register_response: ServerId = register_to_registry()

    if register_response.server_id == "0":
        frontend_server = start_frontend_server()

    execution_profile_server = start_execution_profiler()
    ping_server = start_ping_server()
    inference_server = start_inference_server(register_response.server_id)

    # server_monitor = ServerMonitor(register_response.server_id)
    # server_monitor.init_monitoring()

    # start_iperf3_server()

    inference_server.wait_for_termination()
    execution_profile_server.wait_for_termination()
    ping_server.wait_for_termination()

    if register_response.server_id == "0":
        frontend_server.wait_for_termination()
    pass


def start_iperf3_server():
    iperf3_port = ConfigReader.ConfigReader().read_int("ports", "IPERF3_PORT")
    subprocess.run(
        ["iperf3", "-s", "-p", f"{iperf3_port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def register_to_registry():

    registry_addr = ConfigReader.ConfigReader().read_str("addresses", "REGISTRY_ADDR")
    registry_port = ConfigReader.ConfigReader().read_int("ports", "REGISTRY_PORT")

    register_stub = RegisterStub(
        grpc.insecure_channel("{}:{}".format(registry_addr, registry_port))
    )
    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)

    assignment_port = ConfigReader.ConfigReader().read_int("ports", "ASSIGNMENT_PORT")
    inference_port = ConfigReader.ConfigReader().read_int("ports", "INFERENCE_PORT")
    ping_port = ConfigReader.ConfigReader().read_int("ports", "PING_PORT")
    reachability_info = ReachabilityInfo(
        ip_address=ip_addr,
        assignment_port=assignment_port,
        inference_port=inference_port,
        ping_port=ping_port,
    )
    register_response = register_stub.register_server(reachability_info)

    return register_response


def start_inference_server(server_id: str):
    inference_port = ConfigReader.ConfigReader().read_int("ports", "INFERENCE_PORT")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inferencer = IntermediateServer(server_id)
    add_InferenceServicer_to_server(inferencer, server)
    server.add_insecure_port(f"[::]:{inference_port}")

    server.start()
    print(f"Inference Server running on port {inference_port}...")
    return server


def start_ping_server():

    ping_port = ConfigReader.ConfigReader().read_int("ports", "PING_PORT")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_PingServicer_to_server(PingServer(), server)

    server.add_insecure_port(f"[::]:{ping_port}")
    server.start()
    print(f"Ping Server running on port {ping_port}...")

    return server


def start_execution_profiler():
    execution_profiler_port = ConfigReader.ConfigReader().read_int(
        "ports", "EXECUTION_PROFILER_PORT"
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_ExecutionProfileServicer_to_server(ExecutionProfileServer(), server)

    server.add_insecure_port(f"[::]:{execution_profiler_port}")
    server.start()
    print(f"Execution Profiler Server running on port {execution_profiler_port}...")

    return server


def start_frontend_server():
    frontend_port = ConfigReader.ConfigReader().read_int("ports", "FRONTEND_PORT")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    add_InferenceServicer_to_server(FrontEndServer(), server)

    server.add_insecure_port(f"[::]:{frontend_port}")
    server.start()
    print(f"Frontend Server running on port {frontend_port}...")

    return server


if __name__ == "__main__":

    ## This is just not to print the warning from gRPC
    ## About forking
    os.environ["GRPC_VERBOSITY"] = "NONE"

    main()
