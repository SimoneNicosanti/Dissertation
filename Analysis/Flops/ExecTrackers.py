import time
from abc import ABC, abstractmethod

import keras


class Tracker(ABC):

    def __init__(self, op: keras.Operation):
        super().__init__()
        self.opName = op.name
        self.originalCall = op.call

    @abstractmethod
    def track(self, inputs, *args, **kwargs):
        pass


class ShapeTracker(Tracker):

    def __init__(self, op: keras.Operation):
        super().__init__(op)
        self.trackedShape = None

    def track(self, inputs, *args, **kwargs):
        self.trackedShape = [inputs, args, kwargs]
        return self.originalCall(inputs, *args, **kwargs)


class TimeTracker(Tracker):

    def __init__(self, op: keras.Operation):
        super().__init__(op)
        self.operationsTimes = []

    def track(self, inputs, *args, **kwargs):
        start = time.time_ns()
        result = self.originalCall(inputs, *args, **kwargs)
        end = time.time_ns()
        self.operationsTimes.append(end - start)

        return result


def prepareForTrack(model: keras.Model, trackerClass):

    trackers = {}
    for op in model.operations:
        tracker = trackerClass(op)
        trackers[op.name] = tracker
        op.call = tracker.track

    return trackers


def resetAfterTrack(model: keras.Model, trackers: dict[str, Tracker]):

    for op in model.operations:
        opTracker = trackers[op.name]
        op.call = opTracker.originalCall
