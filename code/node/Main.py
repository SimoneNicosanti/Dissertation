from abc import ABC, abstractmethod
import os
import numpy as np
import requests
import json
import pickle
from ai_edge_litert.interpreter import Interpreter
import random
from prettytable import PrettyTable



class Dispatcher(ABC) :
    @abstractmethod
    def dispatch(self, input) :
        pass


class RemoteDispatcher(Dispatcher) :

    def __init__(self, pipelinePart : str) -> None:
        super().__init__()
        self.part = pipelinePart

    def dispatch(self, input : np.ndarray):
        tf_serving_url = os.getenv('TF_SERVING_URL', 'http://tensorflow-serving:8501')
        # Prepare the data to be sent to TensorFlow Serving
        data = {
            "signature_name": self.part,
            "instances": input.tolist()
        }

        ## Make a request to the TensorFlow Serving model
        response = requests.post(f"{tf_serving_url}/v1/models/mobile_net:predict", json = data)
        text = json.loads(response.text)
        predictions = text["predictions"]

        return np.array(predictions, dtype = np.float32)


class LocalDispatcher(Dispatcher) :

    def __init__(self, part : str) -> None:
        super().__init__()
        interpreter = Interpreter(f"/models/mobile_net.tflite")
        runner = interpreter.get_signature_runner(part)
        self.runner = runner


    def dispatch(self, input : np.ndarray):
        x = input.astype(dtype = np.float32)
        output = self.runner(x = x)
        return output["output_0"]


def main() :
    random.seed(0)

    testElem = readTestElem()

    remoteDispatchers = {"full" : RemoteDispatcher("full")}
    localDispatcher = {"full" : LocalDispatcher("full")}

    result = runPipeline(["full"], [remoteDispatchers], [testElem])
    print(f"Remote Result >>> Max Idx = {result[0][0]} // Max Value = {result[0][1]}")

    result = runPipeline(["full"], [localDispatcher], [testElem])
    print(f"Local Result >>> Max Idx = {result[0][0]} // Max Value = {result[0][1]}")

def readTestElem() :
    testElem = None
    with open("/data/boef_pre.pkl", "rb") as f :
        testElem = pickle.load(f)
    
    return np.array(testElem)

def runPipeline(pipeline, dictList : list[dict[str, Dispatcher]], testList : list[np.ndarray]) :
    predictions = []
    for i in range(len(testList)) :
        input = testList[i]
        for part in pipeline :
            dispatcher = random.choice(dictList)[part]
            output = dispatcher.dispatch(input)
            input = output
        
        predictions.append((np.argmax(output), np.max(output)))
    return predictions



if __name__ == "__main__" :
    main()