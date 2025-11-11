import abc
import time

from readerwriterlock import rwlock

from CommonIds.ComponentId import ComponentId
from CommonPlan.Plan import Plan
from Server.Utils.InferenceInfo import (
    RequestInfo,
    TensorWrapper,
)
from Server.Utils.InputPool import InputPool


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
        print("Request Idx >> ", request_info.request_idx)
        print("\t Component Id >> ", component_id)
        with self.pool_lock.gen_wlock():
            for tensor_wrap in tensor_wrap_list:
                self.input_pool.put_input(component_id, request_info, tensor_wrap)

            input_list, is_ready = self.input_pool.get_input_if_ready(
                component_id, request_info, self.component_input_dict[component_id]
            )

        if is_ready:
            print("\t Is READY to Run Infer")
            start = time.perf_counter_ns()
            infer_res = self.do_inference(component_id, input_list)
            end = time.perf_counter_ns()
            infer_time = (end - start) * 1e-9
            print("\t Run Done with Time >> ", infer_time)
            return infer_res
        else:
            print("\t Is NOT READY to Run Infer")
        return None

    @abc.abstractmethod
    def do_inference(
        self,
        component_id: ComponentId,
        tensor_wrapper_list: list[TensorWrapper],
    ) -> list[TensorWrapper]:
        pass
