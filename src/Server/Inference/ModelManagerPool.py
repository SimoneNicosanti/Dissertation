import threading
from multiprocessing import Process

from Inference import WorkerProcess
from Inference.InferenceInfo import (
    ComponentInfo,
    ModelInfo,
    RequestInfo,
    SharedTensorInfo,
)
from Wrapper.PlanWrapper import PlanWrapper
from Wrapper.QueueWrapper import QueueWrapper


class ModelManagerPool:
    def __init__(self):

        self.dict_lock = threading.Lock()
        self.model_dict: dict[ModelInfo, QueueWrapper] = {}

    def pass_input(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        shared_tensor_info: SharedTensorInfo,
    ):
        with self.dict_lock:
            model_info = component_info.model_info
            if model_info not in self.model_dict:
                raise Exception("Model not found")

            comm_queues = self.model_dict[model_info]
            comm_queues.pass_input(component_info, request_info, shared_tensor_info)

    def spawn_model_component(
        self,
        component_info: ComponentInfo,
        component_path: str,
        plan_wrapper: PlanWrapper,
    ):

        with self.dict_lock:
            model_info = component_info.model_info
            if model_info not in self.model_dict:
                queue_wrapper = QueueWrapper()
                self.model_dict[model_info] = queue_wrapper
                new_model_process = Process(
                    target=WorkerProcess.work,
                    args=(queue_wrapper, plan_wrapper),
                )
                new_model_process.start()
                print(
                    "New Model Process Started for Model {}".format(
                        model_info.model_name
                    )
                )

            queue_wrapper = self.model_dict[model_info]
            queue_wrapper.order_component_spawn(component_info, component_path)
