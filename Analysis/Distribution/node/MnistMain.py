import rpyc
import pickle
import numpy as np


def main() :
    registryConnection = rpyc.connect("localhost", 8000)
    
    firstLayerInfo = registryConnection.root.getLayerHost("preprocess")

    serverConnection = rpyc.connect(firstLayerInfo[0], firstLayerInfo[1])

    testMatrix, trueArray = readTestMatrix()
    print(testMatrix.dtype, testMatrix.shape)
    convertedInput = np.array(testMatrix).tobytes()
    output = serverConnection.root.processLayer("preprocess", convertedInput, testMatrix.shape, testMatrix.dtype.name, 0)
    predictions = np.frombuffer(output[0], dtype=output[2]).reshape(output[1])
    for i in range(predictions.shape[0]) :
        print(f"Elem {i} - Predicted Class >>> {np.argmax(predictions[i])} // True Class >>> {trueArray[i]}")


def readTestMatrix() :
    testMatrix = []
    trueArray = []
    with open("test.pkl", "rb") as f :
        testData = pickle.load(f)
    for row in testData :
        testMatrix.append(row[0])
        trueArray.append(row[1])
    
    return np.array(testMatrix), trueArray





if __name__ == "__main__" :
    main()