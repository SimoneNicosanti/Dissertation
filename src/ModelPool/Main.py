POOL_SERVER_PORT = 50000
from concurrent import futures

import grpc
from PoolServer import PoolServer
from proto.pool_pb2_grpc import add_ModelPoolServicer_to_server


def main():

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pool_server = PoolServer()
    add_ModelPoolServicer_to_server(pool_server, server)
    server.add_insecure_port(f"[::]:{POOL_SERVER_PORT}")
    print("Starting Model Pool Server...")
    server.start()
    print(f"gRPC Model Pool Server running on port {POOL_SERVER_PORT}...")

    server.wait_for_termination()

    pass


if __name__ == "__main__":
    main()
