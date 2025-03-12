import threading

from CommonServer.InferenceInfo import (
    ComponentInfo,
    TensorWrapper,
)
from CommonServer.ModelManager import ModelManager
from CommonServer.PlanWrapper import PlanWrapper
from Server.Inference.ModelRunner import ModelRunner


class IntermediateModelManager(ModelManager):
    def __init__(
        self,
        plan_wrapper: PlanWrapper,
        components_dict: dict[ComponentInfo, str],
        threads_per_model: int,
    ):
        super().__init__(plan_wrapper, components_dict)

        self.execution_semaphore = threading.Semaphore(threads_per_model)

        self.runner_lock_list = [threading.Lock() for _ in range(threads_per_model)]
        self.model_runners = [
            ModelRunner(components_dict) for _ in range(threads_per_model)
        ]

    def do_inference(
        self,
        component_info: ComponentInfo,
        tensor_wrapper_list: list[TensorWrapper],
    ):

        with self.execution_semaphore:
            print("Taken Lock")
            for idx, runner_lock in enumerate(self.runner_lock_list):
                if runner_lock.acquire(blocking=False):
                    ## Run Inference
                    print("Taken Second Lock")
                    model_runner = self.model_runners[idx]
                    output_info_list = model_runner.run_component(
                        component_info, tensor_wrapper_list
                    )
                    print("Inference Done")

                    runner_lock.release()

                    return output_info_list
