import abc

from readerwriterlock import rwlock

from CommonIds.ComponentId import ComponentId
from CommonPlan.Plan import Plan
from CommonServer.InferenceInfo import (
    RequestInfo,
    TensorWrapper,
)
from CommonServer.InputPool import InputPool


class InferenceManager(abc.ABC):
    def __init__(
        self,
        plan: Plan,
        components_dict: dict[ComponentId, str],
    ):

        self.pool_lock = rwlock.RWLockWriteD()
        self.input_pool = InputPool()

        self.component_input_dict: dict[ComponentId, list] = {
            component_info: plan.get_input_names_for_component(component_info)
            for component_info in components_dict
        }

    def pass_input_and_infer(
        self,
        component_id: ComponentId,
        request_info: RequestInfo,
        tensor_wrap_list: list[TensorWrapper],
    ):

        with self.pool_lock.gen_wlock():
            for tensor_wrap in tensor_wrap_list:
                self.input_pool.put_input(component_id, request_info, tensor_wrap)
            print("Put Input")

            input_list, is_ready = self.input_pool.get_input_if_ready(
                component_id, request_info, self.component_input_dict[component_id]
            )
            print("Got Input")

        if is_ready:
            print("Is Ready")
            return self.do_inference(component_id, input_list)
        else:
            print("Not Ready")
        return None

    @abc.abstractmethod
    def do_inference(
        self,
        component_id: ComponentId,
        tensor_wrapper_list: list[TensorWrapper],
    ):
        pass
