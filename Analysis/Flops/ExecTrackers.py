import time
from abc import ABC, abstractmethod

import keras
import tensorflow as tf
from Manipulation import Utils
from Manipulation.NodeWrapper import NodeKey
from tensorflow.python.profiler import model_analyzer, option_builder


class Tracker(ABC):

    def __init__(self, op: keras.Operation):
        super().__init__()
        self.op = op
        self.opName = op.name
        self.originalCall = op.call

        self.totalCalls: int = len(op._inbound_nodes)
        self.nodeIdx = 0

    @abstractmethod
    def track(self, inputs, *args, **kwargs):
        pass

    def updateIdx(self):
        self.nodeIdx += 1
        self.nodeIdx = self.nodeIdx % self.totalCalls

    @abstractmethod
    def getTrackedFromIndex(self, idx: int):
        pass


class ShapeTracker(Tracker):

    def __init__(self, op: keras.Operation):
        super().__init__(op)
        self.inputShapes: dict[int, list] = {x: [] for x in range(0, self.totalCalls)}
        self.outputShapes: dict[int, list] = {x: [] for x in range(0, self.totalCalls)}

    def track(self, inputs, *args, **kwargs):
        ## Assuming main maodel
        nodeKey: NodeKey = NodeKey(None, self.opName, self.nodeIdx)
        self.inputShapes[nodeKey] = [inputs, args, kwargs]
        self.updateIdx()
        return self.originalCall(inputs, *args, **kwargs)

    def getTrackedFromIndex(self, idx):
        return self.inputShapes[idx], self.outputShapes[idx]


class FloatOpsTracker(Tracker):
    def __init__(self, op: keras.Operation):
        super().__init__(op)
        self.floatOpsDict: dict[int, float] = {x: 0 for x in range(0, self.totalCalls)}
        self.unsopported = False

    def track(self, inputs, *args, **kwargs):
        try:
            if isinstance(self.op, keras.layers.InputLayer):
                computedFlops = 0
            else:
                func = tf.function(lambda x: self.originalCall(x[0], *x[1], **x[2]))
                concrete_func = func.get_concrete_function([inputs, args, kwargs])
                graph = concrete_func.graph

                ## Profiling
                opts = option_builder.ProfileOptionBuilder.float_operation()
                graph_info = model_analyzer.profile(graph, options=opts)

                computedFlops = graph_info.total_float_ops

        except TypeError:
            self.unsopported = True
            computedFlops = 0

        self.floatOpsDict[self.nodeIdx] = computedFlops
        self.updateIdx()

        return self.originalCall(inputs, *args, **kwargs)

    def getTrackedFromIndex(self, idx):
        return self.floatOpsDict[idx]


class TimeTracker(Tracker):

    def __init__(self, op: keras.Operation):
        super().__init__(op)
        self.operationsTimes: dict[int, list] = {
            x: [] for x in range(0, self.totalCalls)
        }

    def track(self, inputs, *args, **kwargs):
        start = time.time_ns()
        result = self.originalCall(inputs, *args, **kwargs)
        end = time.time_ns()

        totalTime = end - start
        self.operationsTimes[self.nodeIdx].append(totalTime)

        self.updateIdx()

        return result

    def getTrackedFromIndex(self, idx):
        if isinstance(self.op, keras.layers.InputLayer):
            return [0]
        return self.operationsTimes[idx]


def prepareForTrack(model: keras.Model, trackerClass):

    trackers = {}
    for op in Utils.getModelOperations(model):
        tracker: Tracker = trackerClass(op)
        trackers[op.name] = tracker
        op.call = tracker.track

    return trackers


def resetAfterTrack(model: keras.Model, trackers: dict[str, Tracker]):

    for op in model.operations:
        opTracker = trackers[op.name]
        op.call = opTracker.originalCall
