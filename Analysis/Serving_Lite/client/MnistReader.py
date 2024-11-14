import os
import requests
import json
import keras
import pickle

(trainSet, trainLabels), (testSet, testLabels) = keras.datasets.mnist.load_data()

matrix = []
for i in range(0, 10) :
    matrix.append((testSet[i], testLabels[i]))

# Open a file and use dump() 
with open('./client/test.pkl', 'wb') as file: 
    pickle.dump(matrix, file) 