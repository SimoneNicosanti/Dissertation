from concurrent import futures

import grpc
import keras
from ModelManager import ModelManager
from proto import registry_pb2_grpc
from Registry import Registry


def main():
    model: keras.Model = keras.applications.MobileNetV3Large()
    modelManager: ModelManager = ModelManager(model)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    registry_pb2_grpc.add_RegisterServicer_to_server(Registry(modelManager), server)
    server.add_insecure_port("[::]:5000")
    server.start()
    print("Registry Started")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
