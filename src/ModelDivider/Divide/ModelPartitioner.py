import abc

from CommonPlan.Plan import Plan


class ModelPartitioner(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def partition_model(self, model_plan: Plan, model):
        pass
