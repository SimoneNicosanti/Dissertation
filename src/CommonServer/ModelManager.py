import abc
import threading

from CommonServer.InferenceInfo import (
    ComponentInfo,
    RequestInfo,
    TensorWrapper,
)
from CommonServer.InputPool import InputPool
from CommonServer.PlanWrapper import PlanWrapper


class ModelManager(abc.ABC):
    def __init__(
        self,
        plan_wrapper: PlanWrapper,
        components_dict: dict[ComponentInfo, str],
    ):

        self.pool_lock = threading.Lock()
        self.input_pool = InputPool()

        self.component_input_dict: dict[ComponentInfo, list] = {
            component_info: plan_wrapper.get_input_for_component(component_info)
            for component_info in components_dict
        }

    def pass_input_and_infer(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        input_tensor_info: TensorWrapper,
    ):

        with self.pool_lock:
            self.input_pool.put_input(component_info, request_info, input_tensor_info)
            print("Extracted Input")
            print(self.component_input_dict[component_info])
            input_list, is_ready = self.input_pool.get_input_if_ready(
                component_info, request_info, self.component_input_dict[component_info]
            )

        if is_ready:
            print("Is Ready")
            return self.do_inference(component_info, input_list)

        return None

    @abc.abstractmethod
    def do_inference(
        self,
        component_info: ComponentInfo,
        tensor_wrapper_list: list[TensorWrapper],
    ):
        pass
