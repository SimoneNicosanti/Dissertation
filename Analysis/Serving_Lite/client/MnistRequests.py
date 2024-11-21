import json
import os
import pickle
import random
from abc import ABC, abstractmethod

import numpy as np
import requests
from ai_edge_litert.interpreter import Interpreter
from prettytable import PrettyTable


class Dispatcher(ABC):
    @abstractmethod
    def dispatch(self, input):
        pass


class RemoteDispatcher(Dispatcher):

    def __init__(self, pipelinePart: str) -> None:
        super().__init__()
        self.part = pipelinePart

    def dispatch(self, input: np.ndarray):
        tf_serving_url = os.getenv("TF_SERVING_URL", "http://tensorflow-serving:8501")
        # Prepare the data to be sent to TensorFlow Serving
        data = {"signature_name": self.part, "instances": input.tolist()}

        ## Make a request to the TensorFlow Serving model
        response = requests.post(f"{tf_serving_url}/v1/models/mnist:predict", json=data)
        text = json.loads(response.text)
        predictions = text["predictions"]

        return np.array(predictions, dtype=np.float32)


class LocalDispatcher(Dispatcher):

    def __init__(self, part: str) -> None:
        super().__init__()
        interpreter = Interpreter("/models/mnist.tflite")
        runner = interpreter.get_signature_runner(part)
        self.runner = runner

    def dispatch(self, input: np.ndarray):
        x = input.astype(dtype=np.float32)
        output = self.runner(x=x)
        return output["output_0"]


def main():
    random.seed(0)

    table = PrettyTable()

    testMatrix, trueArray = readTestMatrix()

    pipeline = ["full", "preprocess", "first", "second"]
    remoteDispatchers = {}
    localDispatchers = {}
    for part in pipeline:
        remoteDispatchers[part] = RemoteDispatcher(part)
        localDispatchers[part] = LocalDispatcher(part)

    remotePred = runPipeline(pipeline[1:], [remoteDispatchers], testMatrix[0:])
    localPred = runPipeline(pipeline[1:], [localDispatchers], testMatrix[0:])
    mixedPred = runPipeline(
        pipeline[1:], [localDispatchers, remoteDispatchers], testMatrix[0:]
    )

    table.add_column("Remote", remotePred)
    table.add_column("Local", localPred)
    table.add_column("Mixed", mixedPred)
    table.add_column("True", trueArray)

    print(table)


def readTestMatrix():
    testMatrix = []
    trueArray = []
    with open("test.pkl", "rb") as f:
        testData = pickle.load(f)
    for row in testData:
        testMatrix.append(row[0])
        trueArray.append(row[1])

    return testMatrix, trueArray


def runPipeline(
    pipeline, dictList: list[dict[str, Dispatcher]], testList: list[np.ndarray]
):
    predictions = []
    for i in range(len(testList)):
        input = testList[i]
        for part in pipeline:
            dispatcher = random.choice(dictList)[part]
            output = dispatcher.dispatch(input)
            input = output

        predictions.append(np.argmax(output))
    return predictions


if __name__ == "__main__":
    main()
