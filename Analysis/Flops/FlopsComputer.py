import time

import keras
import numpy as np
import tensorflow as tf
from Flops.ExecTrackers import FloatOpsTracker, ShapeTracker, TimeTracker, Tracker
from Manipulation import Utils
from Manipulation.NodeWrapper import NodeKey, NodePool
from tensorflow.python.profiler import model_analyzer, option_builder


class FlopsComputer:

    def __init__(self, model: keras.Model):
        self.model: keras.Model = model
        self.nodePool: NodePool = NodePool(model)
        self.modelOps: list[keras.Operation] = Utils.getModelOperations(model)

        self.modelNodeKeys: list[NodeKey] = self.nodePool.findModelNodesKeys(None)

    def prepareInputsFromShapes(self, inputShapes: list[tuple]):
        inputs = []
        for inpShape in inputShapes:
            input = tf.random.uniform(shape=inpShape)
            inputs.append(input)

        return inputs

    def computeFloatOpsPerModel(self, inputShapes: dict[str, tuple]) -> float:
        inputSignature = [
            tf.TensorSpec(shape=inputShapes[idx], name=inp.name)
            for idx, inp in enumerate(self.model.inputs)
        ]
        concrete_func = tf.function(self.model).get_concrete_function(inputSignature)
        graph = concrete_func.graph

        # Profile the graph to calculate FLOPs
        opts = option_builder.ProfileOptionBuilder.float_operation()
        graph_info = model_analyzer.profile(graph, options=opts)
        flops = graph_info.total_float_ops

        return flops

    def computeFloatOpsPerOp(self, inputShapes: list[tuple]) -> dict[NodeKey, float]:
        floatOpsTrackers: dict[str, FloatOpsTracker] = self.prepareForTrack(
            FloatOpsTracker
        )

        inputs = self.prepareInputsFromShapes(inputShapes)
        self.model(inputs)

        self.resetAfterTrack(floatOpsTrackers)

        opsNumberDict: dict[NodeKey, float] = {}
        for key in self.modelNodeKeys:
            floatOpsTracker: FloatOpsTracker = floatOpsTrackers.get(key.getOpName())
            floatOps: float = floatOpsTracker.getTrackedFromIndex(key.getOpIdx())
            opsNumberDict[key] = floatOps

        return opsNumberDict

    ## Returns dict of operationName to avgExecutionTime
    def computeRunningTimes(
        self, inputShapes: list[tuple], testNums: int
    ) -> tuple[float, dict[NodeKey, float]]:

        inputs = inputs = self.prepareInputsFromShapes(inputShapes)

        timeTrackers: dict[str, TimeTracker] = self.prepareForTrack(TimeTracker)

        modelTimes = []
        for i in range(testNums):
            start = time.time_ns()
            self.model(inputs)
            end = time.time_ns()
            modelTimes.append(end - start)
            print(f"Time For Call Number {i} >> {end - start} ns")

        self.resetAfterTrack(timeTrackers)

        avgTimes: dict[NodeKey, float] = {}

        for key in self.modelNodeKeys:
            timeTracker: TimeTracker = timeTrackers.get(key.getOpName())
            avgTimes[key] = np.mean(timeTracker.getTrackedFromIndex(key.getOpIdx()))

        modelAvgTime = np.mean(modelTimes)

        return modelAvgTime, avgTimes

    def computeOutputShapes(self, inputShapes: list[tuple]):
        shapeTrackers: dict[str, ShapeTracker] = self.prepareForTrack(ShapeTracker)

        inputs = self.prepareInputsFromShapes(inputShapes)
        self.model(inputs)

        self.resetAfterTrack(shapeTrackers)

        outShapeDict: dict[NodeKey, list[tuple]] = {}
        for key in self.modelNodeKeys:
            shapeTracker: ShapeTracker = shapeTrackers.get(key.getOpName())
            outShapeDict[key] = shapeTracker.getTrackedFromIndex(key.getOpIdx())

        return outShapeDict

    def prepareForTrack(self, trackerClass):
        trackers: dict[str, Tracker] = {}
        for op in Utils.getModelOperations(self.model):
            op: keras.Operation
            tracker: Tracker = trackerClass(op)
            trackers[op.name] = tracker
            op.call = tracker.track

        return trackers

    def resetAfterTrack(self, trackers: dict[str, Tracker]):
        for op in Utils.getModelOperations(self.model):
            op: keras.Operation
            opTracker = trackers[op.name]
            op.call = opTracker.originalCall
