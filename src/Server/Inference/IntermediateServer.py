from CommonServer.InferenceInfo import ComponentInfo, ModelInfo, SharedTensorInfo
from CommonServer.InferenceServer import InferenceServer
from CommonServer.PlanWrapper import PlanWrapper
from Server.Inference.ModelManager import ModelManager


class IntermediateServer(InferenceServer):
    def __init__(self):
        super().__init__()
        self.model_managers: dict[ModelInfo, ModelManager] = {}

    def __do_inference(
        self, component_info, request_info, shared_input_tensor_info
    ) -> list[SharedTensorInfo] | None:

        model_manager = self.model_managers[component_info.model_info]

        shared_output_tensor_info = model_manager.pass_input_and_infer(
            component_info, request_info, shared_input_tensor_info
        )

        return shared_output_tensor_info

    def register_model(
        self,
        model_info: ModelInfo,
        plan_wrapper: PlanWrapper,
        components_dict: dict[ComponentInfo, str],
        processes_per_model: int,
    ):

        with self.lock:
            if model_info not in self.model_managers:
                self.model_managers[model_info] = ModelManager(
                    plan_wrapper, components_dict, processes_per_model
                )
            if model_info not in self.plan_wrapper_dict:
                self.plan_wrapper_dict[model_info] = plan_wrapper
