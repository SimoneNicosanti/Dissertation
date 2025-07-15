from CommonIds.ComponentId import ComponentId
from Server.Utils.InferenceInfo import TensorWrapper
from Server.Utils.InferenceManager import InferenceManager


class ExtremeInferenceManager(InferenceManager):

    def __init__(self, plan_wrapper, components_dict):
        super().__init__(plan_wrapper, components_dict)

    def do_inference(
        self, component_info: ComponentId, tensor_wrapper_list: list[TensorWrapper]
    ):
        return tensor_wrapper_list
