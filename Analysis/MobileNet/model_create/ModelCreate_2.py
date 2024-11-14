import tensorflow as tf
import keras
import functools
import numpy as np
from imageio import imread
import pickle
from keras.applications.mobilenet_v2 import preprocess_input

import networkx as nx
import matplotlib.pyplot as plt

BLOCK_SIZE = 10

class NetworkNode() :

    def __init__(self, layer : keras.Layer) -> None:
        self.layer : keras.Layer = layer
        self.level = None

class NetworkGraph() :
    def __init__(self) -> None:
        self.nodes : list[NetworkNode] = []
        self.archs : dict[str, str] = {}

    def addAdj(self, node_1 : NetworkNode, node_2 : NetworkNode) -> None :
        
        pass

        
    def bfs(self, startNode) -> None:
        queue : list[NetworkNode] = []
        queue.append(startNode)
        currLevel = 0

        while queue :
            currNode : NetworkNode = queue.pop(0)
            if currNode.level == None :
                currNode.level = currLevel
            
            if len(currNode.layer._outbound_nodes) == 1 and len(queue) == 0 :
                currLevel += 1

            

def main():
    data = np.empty((1, 224, 224, 3))
    data[0] = imread('../client/boef.jpg')
    data = preprocess_input(data)
    with open('../client/boef_pre.pkl', "wb") as f :
        pickle.dump(data, f)

    model : keras.Model = keras.applications.MobileNetV2()

    # Create a directed graph
    modelGraph = nx.DiGraph()
    for layer in model.layers:
        modelGraph.add_node(layer.name, level=-1)

    for layer in model.layers:
        inputList = []
        if isinstance(layer.input, list) :
            inputList = layer.input
        else :
            inputList = [layer.input]

        for input in inputList :
            prevLayerInfo = input._keras_history[0] ## Containes info about prev layer
            modelGraph.add_edge(prevLayerInfo.name, layer.name)
    




if __name__ == "__main__":
    main()
