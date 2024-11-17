import rpyc
import keras

from Service import LayerService

from rpyc.utils.server import ThreadedServer
from threading import Thread


## Ricerca del prossimo livello della rete
## Algoritmo tipo chord basato su hash --> Ricerca distribuita senza chiedere sempre al registry:
##  * Nome del livello 
##  * Posizione del livello nella rete
##




def main() :
    model : keras.Model = keras.applications.mobilenet_v2.MobileNetV2()

    # print(model.summary())

    nextLayersDict = {}
    for layer in model.layers:
        if layer.name not in nextLayersDict :
            nextLayersDict[layer.name] = []

        if isinstance(layer.input, list) :
            inputList = layer.input
        else :
            inputList = [layer.input]

        for input in inputList :
            prevLayerInfo = input._keras_history[0] ## Containes info about prev layers
            if prevLayerInfo.name not in nextLayersDict :
                nextLayersDict[prevLayerInfo.name] = []
            nextLayersDict[prevLayerInfo.name].append(layer.name)

    c = rpyc.connect("localhost", 8000)
    
    portNum = 9000
    allThreads = []
    MAX_LAYERS_PER_SERVICE = 20
    for i in range(0, len(model.layers), MAX_LAYERS_PER_SERVICE) :
        
        layerSubList = model.layers[i : min(i + MAX_LAYERS_PER_SERVICE, len(model.layers))]

        ## Subsribing Layer with a service
        for layer in layerSubList :
            c.root.subscribeLayer(layer.name, "localhost", portNum)
        
        serverThread = Thread(target = startServer, args = (layerSubList, nextLayersDict, portNum), daemon = True)
        serverThread.start()
        allThreads.append(serverThread)
        portNum += 1
    
    for thread in allThreads :
        thread.join()


def startServer(layerList, nextLayersDict, portNum) :
    t = ThreadedServer(
        LayerService(layerList, nextLayersDict), 
        port = portNum
    )
    print("Starting Service")
    t.start()

        
if __name__ == "__main__" :
    main()