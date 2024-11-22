import pickle
import time

import grpc
import keras
import numpy as np
import tensorflow as tf
from proto import registry_pb2_grpc, server_pb2_grpc
from proto.registry_pb2 import LayerInfo, LayerPosition
from proto.server_pb2 import LayerRequest, LayerResponse


def remoteTest():
    inputTensor = readTestElem()
    with grpc.insecure_channel("localhost:5000") as channel:
        stub = registry_pb2_grpc.RegisterStub(channel)
        layerPosition: LayerPosition = stub.getLayerPosition(
            LayerInfo(modelName="", layerName="input_layer")
        )
        print(f"Received >>> {layerPosition.layerHost}:{layerPosition.layerPort}")

        with grpc.insecure_channel(
            f"{layerPosition.layerHost}:{layerPosition.layerPort}"
        ) as layerChannel:
            serverStub = server_pb2_grpc.ServerStub(layerChannel)
            layerRequest: LayerRequest = LayerRequest(
                modelName="",
                layerName="input_layer",
                requestId=0,
                tensor=tf.make_tensor_proto(inputTensor),
            )
            start = time.time()
            result: LayerResponse = serverStub.serveLayer(layerRequest)
            end = time.time()
            tensorResult = tf.convert_to_tensor(tf.make_ndarray(result.result))
            for i in range(0, len(tensorResult)):
                print(f"Predicted Class >>> {np.argmax(tensorResult[i])}")
            print(f"Remote Time >>> {end - start}")


def localTest():
    testElem = readTestElem()
    model = keras.applications.MobileNetV3Large()
    start = time.time()
    predictions = model(testElem)
    end = time.time()
    print(f"Local Time >>> {end - start}")
    print(f"Predicted >>> {np.argmax(predictions)}")


def readTestElem():
    testElem = None
    with open("boef_pre.pkl", "rb") as f:
        testElem = pickle.load(f)

    return tf.convert_to_tensor(value=testElem)


if __name__ == "__main__":
    remoteTest()
    localTest()
