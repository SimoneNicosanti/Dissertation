import rpyc
import pickle
import numpy as np


def main() :
    registryConnection = rpyc.connect("localhost", 8000)
    
    firstLayerInfo = registryConnection.root.getLayerHost("Conv1")

    serverConnection = rpyc.connect(firstLayerInfo[0], firstLayerInfo[1], config={"sync_request_timeout": None})

    testElem = readTestElem()
    print(testElem.dtype, testElem.shape)
    convertedTestElem = testElem.tobytes()

    output = serverConnection.root.processLayer("Conv1", convertedTestElem, testElem.shape, testElem.dtype.name, 0)
    predictions = np.frombuffer(output[0], dtype=output[2]).reshape(output[1])
    for i in range(predictions.shape[0]) :
        print(f"Elem {i} - Predicted Class >>> {np.argmax(predictions[i])}")


def readTestElem() :
    testElem = None
    with open("boef_pre.pkl", "rb") as f :
        testElem = pickle.load(f)
    
    return np.array([testElem[0], testElem[0], testElem[0]], dtype=np.float64)





if __name__ == "__main__" :
    main()