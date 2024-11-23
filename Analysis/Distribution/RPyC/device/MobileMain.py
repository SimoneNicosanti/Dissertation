import pickle
import time

import keras
import numpy as np
import rpyc

TEST_NUM = 50


def remoteTest():
    registryConnection = rpyc.connect("localhost", 8000)

    firstLayerInfo = registryConnection.root.getLayerHost("input_layer")

    serverConnection = rpyc.connect(
        firstLayerInfo[0], firstLayerInfo[1], config={"sync_request_timeout": None}
    )

    testElem = readTestElem()
    print(testElem.dtype, testElem.shape)
    convertedTestElem = testElem.tobytes()
    timeArray = np.zeros(shape=TEST_NUM)
    for i in range(0, TEST_NUM):
        print(f"Iteration >>> {i}")
        start = time.time()
        output = serverConnection.root.processLayer(
            "input_layer", convertedTestElem, testElem.shape, testElem.dtype.name, 0
        )
        end = time.time()
        timeArray[i] = end - start
    print(
        f"Avg Remote Time >>> {timeArray.mean()} // Remote Time Std dev {timeArray.std()}"
    )
    predictions = np.frombuffer(output[0], dtype=output[2]).reshape(output[1])
    for i in range(predictions.shape[0]):
        print(f"Elem {i} - Predicted Class >>> {np.argmax(predictions[i])}")


def localTest():
    testElem = readTestElem()
    model = keras.applications.MobileNetV3Large()
    start = time.time()
    predictions = model(testElem)
    end = time.time()
    print(f"Local Time >>> {end - start}")
    print(f"Predicted >>> {np.argmax(predictions)}")


def main():
    remoteTest()
    localTest()


def readTestElem():
    testElem = None
    with open("boef_pre.pkl", "rb") as f:
        testElem = pickle.load(f)

    return np.array([testElem[0]], dtype=np.float64)


if __name__ == "__main__":
    main()
