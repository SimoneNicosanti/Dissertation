import pickle
import time

import grpc
import keras
import numpy as np
import tensorflow as tf
from proto import registry_pb2_grpc, server_pb2_grpc
from proto.registry_pb2 import LayerInfo, ServerInfo
from proto.server_pb2 import ModelInput, ModelOutput

TEST_NUM = 1


def remoteTest():
    inputTensor = readTestElem()
    timeArray = np.zeros(shape=TEST_NUM)
    with grpc.insecure_channel("registry:5000") as channel:
        stub = registry_pb2_grpc.RegisterStub(channel)
        layerPosition: ServerInfo = stub.getLayerPosition(
            LayerInfo(modelName="", layerName="input_layer_1_0_0")
        )
        print(f"Received >>> {layerPosition.hostName}:{layerPosition.portNum}")
        with grpc.insecure_channel(
            f"{layerPosition.hostName}:{layerPosition.portNum}"
        ) as layerChannel:
            serverStub = server_pb2_grpc.ServerStub(layerChannel)
            for i in range(0, TEST_NUM):
                print(f"Iteration >>> {i}")
                modelInput: ModelInput = ModelInput(
                    modelName="",
                    layerName="input_layer_1_0_0",
                    requestId=0,
                    tensor=tf.make_tensor_proto(inputTensor),
                )
                start = time.time()
                modelOutput: ModelOutput = serverStub.serveModel(modelInput)
                end = time.time()
                timeArray[i] = end - start

                tensorResult = tf.make_ndarray(modelOutput.result["box_0"])
            print(
                f"Avg Remote Time >>> {timeArray.mean()} // Remote Time Std dev {timeArray.std()}"
            )
            return tensorResult


def localTest():
    testElem = readTestElem()
    model = keras.saving.load_model("/models/UnnestedYolo.keras")
    timeArray = np.zeros(shape=TEST_NUM)
    for i in range(0, TEST_NUM):
        start = time.time()
        predictions = model(testElem)
        end = time.time()
        timeArray[i] = end - start
    print(
        f"Avg Local Time >>> {timeArray.mean()} // Local Time Std dev {timeArray.std()}"
    )
    return predictions["box_0"]


def readTestElem():
    testElem = None
    with open("boef_pre.pkl", "rb") as f:
        testElem = pickle.load(f)

    return tf.convert_to_tensor(value=testElem, dtype=tf.float32)


if __name__ == "__main__":
    remRes = remoteTest()
    locRes = localTest()
    print("Are Results Equal >>> ", np.array_equal(remRes, locRes))
    print(f"Norm of Difference >>> {tf.norm(remRes - locRes)}")
