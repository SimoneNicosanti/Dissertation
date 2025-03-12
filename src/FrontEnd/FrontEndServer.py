import time

from CommonServer.InferenceInfo import (
    ComponentInfo,
    ModelInfo,
    RequestInfo,
    SharedTensorInfo,
)
from CommonServer.InferenceServer import InferenceServer
from CommonServer.PlanWrapper import PlanWrapper

## This has to be a different service running on the client
## Actually it should be started by the deployer --> I.E. The one asking for the plan optimization


class FrontEndServer(InferenceServer):
    def __init__(self) -> None:
        super().__init__()
        self.pending_request_dict = {}
        pass

    def __do_inference(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        shared_input_tensor_info: SharedTensorInfo,
    ):
        plan_wrapper = self.plan_dict[component_info.model_info]
        if plan_wrapper.is_only_input_component(component_info):
            self.pending_request_dict[request_info] = time.time_ns()
            return shared_input_tensor_info

        elif plan_wrapper.is_only_output_component(component_info):
            start = self.pending_request_dict.pop(request_info)
            print("Total Time >> ", time.time_ns() - start)
            return None

        return None

    def register_model(
        self,
        model_info: ModelInfo,
        plan_wrapper: PlanWrapper,
    ):
        with self.lock:
            self.plan_wrapper_dict[model_info] = plan_wrapper
