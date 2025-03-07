import socket
from concurrent import futures

import grpc
from Assignee.Assignee import Fetcher
from proto.server_pb2_grpc import add_AssigneeServicer_to_server
from proto.register_pb2 import ReachabilityInfo, RegisterResponse
from proto.register_pb2_grpc import RegisterStub

ASSIGNEE_PORT = 50040
INFERENCE_PORT = 50030
PING_PORT = 50020


def main():

    ## Register to Registry
    ## Start Assignee

    register_response: RegisterResponse = register_to_registry()
    assignee_server = start_assignee_server(register_response.server_id)
    # inference_server = start_inference_server()

    assignee_server.wait_for_termination()
    # inference_server.wait_for_termination()
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


def start_assignee_server(server_id: int):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fetcher = Fetcher(server_id, "/models", None)
    add_AssigneeServicer_to_server(fetcher, server)
    server.add_insecure_port(f"[::]:{ASSIGNEE_PORT}")
    print(f"Assignee Server running on port {ASSIGNEE_PORT}...")

    server.start()
    return server


def start_inference_server(server_id: int):
    ## TODO Complete this
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server.start()
    return server


if __name__ == "__main__":
    main()
