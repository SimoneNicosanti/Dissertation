import rpyc
import keras
import numpy as np

class RequestPool() :

    def __init__(self) :
        self.requestMap : dict[tuple[str, int], list] = {} ## Maps (Layer Name, requestId) --> list of inputs
    
    def addRequest(self, layerName, requestId, input) :
        key = (layerName, requestId)
        if key not in self.requestMap :
            self.requestMap[key] = []
        self.requestMap[key].append(input)
        
    def getRequestInfo(self, layerName, requestId) :
        key = (layerName, requestId)
        return self.requestMap[key]

    def removeRequestInfo(self, layerName, requestId) :
        key = (layerName, requestId)
        self.requestMap.pop(key)


class LayerService(rpyc.Service):

    requestPool = RequestPool()

    def __init__(self, layers : list[keras.Layer], nextLayersDict : dict[str, list[str]]) :
        super().__init__()
        self.layerDict : dict[str, keras.Layer] = {}
        for layer in layers :
            self.layerDict[layer.name] = layer
        self.nextLayersDict : dict[str, list[str]] = nextLayersDict

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_processLayer(self, layerName : str, input : np.ndarray, requestId : int) : # this is an exposed method
        
        convertedInput = np.array(input)
        currLayer : keras.Layer = self.layerDict[layerName]

        LayerService.requestPool.addRequest(currLayer.name, requestId, convertedInput)

        if isinstance(currLayer.input, list) :
            inputList = currLayer.input
        else :
            inputList = [currLayer.input]
        
        collectedInputs = LayerService.requestPool.getRequestInfo(currLayer.name, requestId)
        if len(inputList) == len(collectedInputs) :
            layerResult = currLayer(collectedInputs)
            LayerService.requestPool.removeRequestInfo(currLayer.name, requestId)

            nextLayers = self.nextLayersDict[layerName]
            if (len(nextLayers) == 0) :
                ## This is the last layer of the network
                return layerResult
            
            
            c = rpyc.connect("localhost", 8000)
            for nextLayerName in nextLayers :
                nextLayerHost : tuple = c.root.getLayerHost(nextLayerName)
                nextLayerConnection = rpyc.connect(nextLayerHost[0], nextLayerHost[1])
                returnedValue = nextLayerConnection.root.processLayer(nextLayerName, layerResult.numpy().tolist(), requestId) ## Gestire caso multi input

                if returnedValue != None :
                    ## Next Layer has all its inputs and has returned a valid value
                    return returnedValue
        else :
            return None
            
            
        