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

        self.next_runner_idx_lock = threading.Lock()
        self.next_runner_idx = 0

        self.model_runners = [
            ModelRunner(components_dict) for _ in range(threads_per_model)
        ]

    def do_inference(
        self,
        component_info: ComponentInfo,
        tensor_wrapper_list: list[TensorWrapper],
    ):
        ## Firt of all we have to take the semaphore
        ## This ensures that there will be at least one free model runner
        # with self.execution_semaphore:

        ## Taking the next runner --> Round robin politic
        # with self.next_runner_idx_lock:
        idx = self.next_runner_idx
        self.next_runner_idx = (self.next_runner_idx + 1) % len(self.runner_lock_list)

        ## Taking the lock on that runner and running inference
        # with self.runner_lock_list[idx]:
        ## Run Inference
        model_runner = self.model_runners[idx]
        output_info_list = model_runner.run_component(
            component_info, tensor_wrapper_list
        )
        print("Inference Done")

        return output_info_list

        # for idx, runner_lock in enumerate(self.runner_lock_list):
        #     ## Then we have to find a free runner taking the lock
        #     if runner_lock.acquire(blocking=False):
        #         ## Run Inference
        #         model_runner = self.model_runners[idx]
        #         output_info_list = model_runner.run_component(
        #             component_info, tensor_wrapper_list
        #         )
        #         print("Inference Done")

        #         runner_lock.release()

        #         return output_info_list
