import rpyc
import pickle
import numpy as np


def main() :
    registryConnection = rpyc.connect("localhost", 8000)
    
    firstLayerInfo = registryConnection.root.getLayerHost("preprocess")

    serverConnection = rpyc.connect(firstLayerInfo[0], firstLayerInfo[1])

    testMatrix, trueArray = readTestMatrix()

    print(serverConnection.root.processLayer("preprocess", np.array([testMatrix[0]]).tolist(), 0))


def readTestMatrix() :
    testMatrix = []
    trueArray = []
    with open("test.pkl", "rb") as f :
        testData = pickle.load(f)
    for row in testData :
        testMatrix.append(row[0])
        trueArray.append(row[1])
    
    return testMatrix, trueArray





if __name__ == "__main__" :
    main()