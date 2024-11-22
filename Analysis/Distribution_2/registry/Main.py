from concurrent import futures

import grpc
from proto import registry_pb2_grpc
from Registry import Registry


def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    registry_pb2_grpc.add_RegisterServicer_to_server(Registry(), server)
    server.add_insecure_port("[::]:5000")
    server.start()
    print("Registry Started")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
