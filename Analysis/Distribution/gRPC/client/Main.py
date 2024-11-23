import pickle
import time

import grpc
import keras
import numpy as np
import tensorflow as tf
from proto import registry_pb2_grpc, server_pb2_grpc
from proto.registry_pb2 import LayerInfo, LayerPosition
from proto.server_pb2 import LayerRequest, LayerResponse

TEST_NUM = 1


def remoteTest():
    inputTensor = readTestElem()
    timeArray = np.zeros(shape=TEST_NUM)
    with grpc.insecure_channel("registry:5000") as channel:
        stub = registry_pb2_grpc.RegisterStub(channel)
        layerPosition: LayerPosition = stub.getLayerPosition(
            LayerInfo(modelName="", layerName="input_layer")
        )
        print(f"Received >>> {layerPosition.layerHost}:{layerPosition.layerPort}")

        with grpc.insecure_channel(
            f"{layerPosition.layerHost}:{layerPosition.layerPort}"
        ) as layerChannel:
            serverStub = server_pb2_grpc.ServerStub(layerChannel)
            for i in range(0, TEST_NUM):
                print(f"Iteration >>> {i}")
                layerRequest: LayerRequest = LayerRequest(
                    modelName="",
                    layerName="input_layer",
                    requestId=0,
                    tensor=tf.make_tensor_proto(inputTensor),
                )
                start = time.time()
                result: LayerResponse = serverStub.serveLayer(layerRequest)
                end = time.time()
                timeArray[i] = end - start
            tensorResult = tf.convert_to_tensor(tf.make_ndarray(result.result))
            for i in range(0, len(tensorResult)):
                print(f"Predicted Class >>> {np.argmax(tensorResult[i])}")
            print(
                f"Avg Remote Time >>> {timeArray.mean()} // Remote Time Std dev {timeArray.std()}"
            )


def localTest():
    testElem = readTestElem()
    model = keras.applications.MobileNetV3Large()
    timeArray = np.zeros(shape=TEST_NUM)
    for i in range(0, TEST_NUM):
        start = time.time()
        predictions = model(testElem)
        end = time.time()
        timeArray[i] = end - start
    print(
        f"Avg Local Time >>> {timeArray.mean()} // Remote Time Std dev {timeArray.std()}"
    )
    print(f"Predicted >>> {np.argmax(predictions)}")


def readTestElem():
    testElem = None
    with open("boef_pre.pkl", "rb") as f:
        testElem = pickle.load(f)

    return tf.convert_to_tensor(value=testElem)


if __name__ == "__main__":
    remoteTest()
    localTest()
