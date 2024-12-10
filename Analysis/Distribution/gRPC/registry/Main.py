from concurrent import futures

import grpc
import keras
from Manipulation import Split
from proto import registry_pb2_grpc
from Registry import Registry


def main():
    # model: keras.Model = keras.applications.MobileNetV3Large()
    model: keras.Model = keras.saving.load_model("/models/UnpackedYolo.keras")

    subModels: list[keras.Model] = Split.modelSplit(model, 100)
    for i, mod in enumerate(subModels):
        mod.save(f"/models/SubModel_{i}.keras")

    print("Model Parts >>> ", len(subModels))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    registry_pb2_grpc.add_RegisterServicer_to_server(
        Registry(partsNum=len(subModels)), server
    )
    server.add_insecure_port("[::]:5000")
    server.start()
    print("Registry Started")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
