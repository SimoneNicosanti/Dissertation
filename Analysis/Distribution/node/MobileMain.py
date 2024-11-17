import rpyc
import pickle
import numpy as np


def main() :
    registryConnection = rpyc.connect("localhost", 8000)
    
    firstLayerInfo = registryConnection.root.getLayerHost("Conv1")

    serverConnection = rpyc.connect(firstLayerInfo[0], firstLayerInfo[1])

    testElem = readTestElem()

    output = serverConnection.root.processLayer("Conv1", testElem.tolist(), 0)
    print(type(output))
    output = np.array(output)
    print(output.shape)


def readTestElem() :
    testElem = None
    with open("boef_pre.pkl", "rb") as f :
        testElem = pickle.load(f)
    
    return np.array(testElem)





if __name__ == "__main__" :
    main()