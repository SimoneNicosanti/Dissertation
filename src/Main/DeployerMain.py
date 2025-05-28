import os
from concurrent import futures

import grpc

from Common import ConfigReader
from Deployer.DeploymentServer import DeploymentServer
from proto_compiled.deployment_pb2_grpc import add_DeploymentServicer_to_server


def main():
    dir_list = ConfigReader.ConfigReader().read_all_dirs("deployment_dirs")
    for dir in dir_list:
        os.makedirs(dir, exist_ok=True)

    deployment_server = start_deployment_server()
    deployment_server.wait_for_termination()

    pass


def start_deployment_server() -> grpc.Server:
    deployment_port = ConfigReader.ConfigReader().read_int("ports", "DEPLOYER_PORT")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_DeploymentServicer_to_server(DeploymentServer(), server)
    server.add_insecure_port(f"[::]:{deployment_port}")

    server.start()
    print(f"gRPC Deployment Server running on port {deployment_port}...")

    return server


if __name__ == "__main__":
    main()
