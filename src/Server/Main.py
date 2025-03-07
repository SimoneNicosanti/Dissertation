from proto.register_pb2_grpc import RegisterStub
from proto.register_pb2 import ReachabilityInfo, RegisterResponse
import grpc
import socket


ASSIGNEE_PORT = 50052
PING_PORT = 50053

def main():

    ## Register to Registry
    ## Start Assignee

    register_response : RegisterResponse = register_to_registry()
    start_assignee(register_response.server_id)

    pass

def register_to_registry() :
    register_stub = RegisterStub(grpc.insecure_channel("registry:50051"))
    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)
    reachability_info = ReachabilityInfo(ip_address=ip_addr, assignment_port=ASSIGNEE_PORT, ping_port=PING_PORT)
    register_response = register_stub.register_server(reachability_info)
    
    return register_response

def start_assignee(server_id : int) :
    pass


if __name__ == "__main__":
    main()