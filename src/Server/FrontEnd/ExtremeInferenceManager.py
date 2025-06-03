from CommonIds.ComponentId import ComponentId
from CommonServer.InferenceInfo import TensorWrapper
from CommonServer.InferenceManager import InferenceManager


class ExtremeInferenceManager(InferenceManager):

    def __init__(self, plan_wrapper, components_dict):
        super().__init__(plan_wrapper, components_dict)

    def do_inference(
        self, component_info: ComponentId, tensor_wrapper_list: list[TensorWrapper]
    ):
        print("Here")
        return tensor_wrapper_list
