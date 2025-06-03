import threading

from CommonIds.ComponentId import ComponentId
from CommonPlan.Plan import Plan
from Server.Inference.ModelRunner import ModelRunner
from Server.Utils.InferenceInfo import (
    TensorWrapper,
)
from Server.Utils.InferenceManager import InferenceManager


class IntermediateInferenceManager(InferenceManager):
    def __init__(
        self,
        plan: Plan,
        components_dict: dict[ComponentId, str],
        threads_per_model: int,
    ):
        super().__init__(plan, components_dict)

        self.execution_semaphore = threading.Semaphore(threads_per_model)
        self.runner_lock_list = [threading.Lock() for _ in range(threads_per_model)]

        self.next_runner_idx_lock = threading.Lock()
        self.next_runner_idx = 0

        self.model_runners = [
            ModelRunner(components_dict) for _ in range(threads_per_model)
        ]

    def do_inference(
        self,
        component_id: ComponentId,
        tensor_wrapper_list: list[TensorWrapper],
    ):
        ## Firt of all we have to take the semaphore
        ## This ensures that there will be at least one free model runner
        with self.execution_semaphore:

            ## Taking the next runner --> Round robin politic
            with self.next_runner_idx_lock:
                idx = self.next_runner_idx
                self.next_runner_idx = (self.next_runner_idx + 1) % len(
                    self.runner_lock_list
                )

            ## Taking the lock on that runner and running inference
            with self.runner_lock_list[idx]:
                ## Run Inference
                model_runner = self.model_runners[idx]
                output_info_list = model_runner.run_component(
                    component_id, tensor_wrapper_list
                )
                print("Inference Done")

        return output_info_list
