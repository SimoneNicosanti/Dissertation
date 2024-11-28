import time

import keras
import ModelParse
import numpy as np
import tensorflow as tf
from tensorflow.python.profiler import model_analyzer, option_builder


def runLayer(
    subMod: keras.Model,
    outputsDict: dict[str, keras.Model],
    results: dict[str],
    executionTimes: dict[str, float],
):
    if len(subMod.input) == 0:
        ## Input Layer
        for outKey in subMod.output:
            results[outKey] = tf.random.uniform(shape=(1, 100, 100, 3))

    else:
        ## Processing Layer
        for inputName in subMod.input:
            if inputName not in results:
                prevSubMod = outputsDict[inputName]
                runLayer(prevSubMod, outputsDict, results)

        aggregatedInput = {key: results[key] for key in subMod.input}
        start = time.time_ns()
        subOutput = subMod(aggregatedInput)
        end = time.time_ns()
        executionTimes[subMod.layers[-1].name].append(end - start)
        for outName in subOutput:
            results[outName] = subOutput[outName]


def computeTimes(subModels, outputsDict):
    executionTimes = {}
    for mod in subModels:
        executionTimes[mod.layers[-1].name] = []

    for i in range(0, 1):
        results = {"input": tf.random.uniform(shape=(1, 32, 32, 3))}
        for subMod in subModels:
            runLayer(subMod, outputsDict, results, executionTimes)

    for key in executionTimes:
        print(f"Avg Time {key} >>> {np.mean(executionTimes[key])}")

    return results, executionTimes


def computeOps(subModels, results):
    operations = {}
    for mod in subModels:
        input_signature = {
            name: tf.TensorSpec(
                shape=results[name].shape, dtype=results[name].dtype, name=name
            )
            for name in mod.input.keys()
        }

        # Wrap the model in a tf.function that explicitly handles dictionary inputs
        @tf.function(input_signature=[input_signature])
        def wrapped_model(inputs):
            return mod(inputs)

        # Bind inputs using results
        inputs = {name: results[name] for name in mod.input.keys()}
        concrete_func = wrapped_model.get_concrete_function(inputs)
        forward_graph = concrete_func.graph

        # Profile the graph to get FLOPs
        options = option_builder.ProfileOptionBuilder.float_operation()
        graph_info = model_analyzer.profile(forward_graph, options=options)
        flops = graph_info.total_float_ops

        # Use the last layer's name to store the FLOPs
        operations[mod.layers[-1].name] = flops

    print(operations)


# 2160000
# 221184


def main():
    model: keras.Model = keras.applications.MobileNetV3Large()
    subModels = ModelParse.modelParse(model, maxLayerNum=1)  ## One layer per sub model
    ## Who has to generate each input tensor
    outputsDict = {}
    for mod in subModels:
        for outKey in mod.output:
            outputsDict[outKey] = mod
    results, times = computeTimes(subModels, outputsDict)
    # print(results.keys())
    computeOps(subModels, results)


if __name__ == "__main__":
    main()
