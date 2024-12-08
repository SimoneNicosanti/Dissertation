import time

import keras
import numpy as np
import tensorflow as tf
from Flops import ExecTrackers
from Flops.ExecTrackers import ShapeTracker, TimeTracker
from tensorflow.python.profiler import model_analyzer, option_builder


def computeFloatOperations(
    model: keras.Model, inputShape: tuple
) -> tuple[float, dict[str, float]]:
    floatOpsPerOp: dict[str, float] = computeFloatOperationsPerOperation(
        model, inputShape
    )
    floatOpsPerModel: float = computeFloatOperationsPerModel(model, inputShape)

    return floatOpsPerModel, floatOpsPerOp


def computeFloatOperationsPerModel(model: keras.Model, inputShape: tuple) -> float:

    inputSignature = [
        tf.TensorSpec(shape=inputShape, name=inp.name) for inp in model.inputs
    ]
    concrete_func = tf.function(model).get_concrete_function(inputSignature)
    graph = concrete_func.graph

    # Profile the graph to calculate FLOPs
    opts = option_builder.ProfileOptionBuilder.float_operation()
    graph_info = model_analyzer.profile(graph, options=opts)
    flops = graph_info.total_float_ops

    return flops


def computeFloatOperationsPerOperation(
    model: keras.Model, inputShape: tuple
) -> dict[str, float]:

    shapeTrackers: dict[str, ShapeTracker] = ExecTrackers.prepareForTracking(
        model, ShapeTracker
    )

    x = tf.random.uniform(shape=inputShape)
    model(x)

    ExecTrackers.resetAfterTrack(model, shapeTrackers)

    opsNumberDict: dict[str, float] = {op.name: 0 for op in model.operations}
    unsopportedOps: list[str] = []

    for idx, _ in enumerate(model.operations):
        op: keras.Operation = model.operations[idx]
        if not isinstance(op, keras.layers.InputLayer):
            inputSignature = shapeTrackers[op.name].trackedShape

            try:
                func = tf.function(lambda x: op.call(x[0], *x[1], **x[2]))
                concrete_func = func.get_concrete_function(inputSignature)

                # Convert the concrete function to a frozen graph
                graph = concrete_func.graph

                # Profile the graph to calculate FLOPs
                opts = option_builder.ProfileOptionBuilder.float_operation()
                graph_info = model_analyzer.profile(graph, options=opts)
                flops = graph_info.total_float_ops

                opsNumberDict[op.name] = flops
            except TypeError:
                print("Unsopported Type Per tf.function >>> Assuming 0 flops")
                opsNumberDict[op.name] = 0.0
                unsopportedOps.append(op.name)

    print(f"TOTAL UNSOPPORTED OPS >>> {len(unsopportedOps)}")
    return opsNumberDict


## Returns dict of operationName to avgExecutionTime
def computeRunningTimes(
    model: keras.Model, inputShape: tuple, testNums: int
) -> tuple[float, dict[str, float]]:

    avgTimes: dict[str, float] = {}

    x = tf.random.uniform(shape=inputShape)

    timeTrackers: dict[str, TimeTracker] = ExecTrackers.prepareForTracking(
        model, TimeTracker
    )

    modelTimes = []
    for i in range(testNums):
        start = time.time_ns()
        model(x)
        end = time.time_ns()
        modelTimes.append(end - start)
        print(f"Time For Call Number {i} >> {end - start} ns")

    ExecTrackers.resetAfterTrack(model, timeTrackers)

    for opName in timeTrackers:
        tracker: TimeTracker = timeTrackers[opName]
        avgTimes[tracker.opName] = np.mean(tracker.operationsTimes)

    modelAvgTime = np.mean(modelTimes)

    return modelAvgTime, avgTimes


def computeFlopsPerOp(
    model: keras.Model, inputShape: tuple, testNums: int
) -> dict[str, tuple[float, float, float]]:

    floatOpsPerOp: dict[str, float] = computeFloatOperationsPerOperation(
        model, inputShape
    )

    _, avgTimesPerOp = computeRunningTimes(model, inputShape, testNums)
    print(avgTimesPerOp)

    flopsPerOp = {}

    for op in model.operations:
        floatOps = floatOpsPerOp.get(op.name, 0)
        avgTime = avgTimesPerOp.get(op.name, 1)
        flops = floatOps / avgTime
        flopsPerOp[op.name] = (floatOps, avgTime, flops)

    return flopsPerOp


## Returns dict of operationName to (Floating Point Operations, AvgTime, FLOPS)
def computeFlopsInfoPerModel(
    model: keras.Model, inputShape: tuple, testNums: int
) -> dict[str, tuple[float, float, float]]:

    modelFloatOps: float = computeFloatOperationsPerModel(model, inputShape)
    modelAvgTime, _ = computeRunningTimes(model, inputShape, testNums)

    return modelFloatOps / modelAvgTime
