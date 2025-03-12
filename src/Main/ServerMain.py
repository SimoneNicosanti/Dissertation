import socket
from concurrent import futures

import grpc

from proto_compiled.register_pb2 import ReachabilityInfo, ServerId
from proto_compiled.register_pb2_grpc import RegisterStub
from proto_compiled.server_pb2_grpc import (
    add_AssigneeServicer_to_server,
    add_InferenceServicer_to_server,
)
from Server.Assignee.Assignee import Fetcher
from Server.Inference.IntermediateServer import IntermediateServer

ASSIGNEE_PORT = 50040
INFERENCE_PORT = 50030
PING_PORT = 50020


def main():

    ## Register to Registry
    ## Start Assignee

    register_response: ServerId = register_to_registry()

    inferencer, inference_server = start_inference_server()
    assignee_server = start_assignee_server(register_response.server_id, inferencer)

    assignee_server.wait_for_termination()
    inference_server.wait_for_termination()
    pass


def register_to_registry():
    register_stub = RegisterStub(grpc.insecure_channel("registry:50051"))
    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)
    reachability_info = ReachabilityInfo(
        ip_address=ip_addr,
        assignment_port=ASSIGNEE_PORT,
        inference_port=INFERENCE_PORT,
        ping_port=PING_PORT,
    )
    register_response = register_stub.register_server(reachability_info)

    return register_response


def start_assignee_server(server_id: int, intermediate_server):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fetcher = Fetcher(server_id, "/models", intermediate_server)
    add_AssigneeServicer_to_server(fetcher, server)
    server.add_insecure_port(f"[::]:{ASSIGNEE_PORT}")
    print(f"Assignee Server running on port {ASSIGNEE_PORT}...")

    server.start()
    return server


def start_inference_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inferencer = IntermediateServer()
    add_InferenceServicer_to_server(inferencer, server)
    server.add_insecure_port(f"[::]:{INFERENCE_PORT}")

    server.start()
    print(f"Inference Server running on port {INFERENCE_PORT}...")
    return inferencer, server


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn")
    main()
