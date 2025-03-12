import multiprocessing

from CommonServer.InferenceInfo import (
    ComponentInfo,
    RequestInfo,
    SharedTensorInfo,
)
from CommonServer.PlanWrapper import PlanWrapper
from Server.Inference import WorkerProcess
from Server.Inference.InputPool import InputPool
from Server.Wrapper.PipeWrapper import PipeWrapper


class ModelManager:
    def __init__(
        self,
        plan_wrapper: PlanWrapper,
        components_dict: dict[ComponentInfo, str],
        processes_per_model: int,
    ):

        self.process_dict: dict[int, PipeWrapper] = {}
        self.__init_processes(processes_per_model, components_dict)

        self.input_pool = InputPool()

        self.component_input_dict: dict[ComponentInfo, list] = [
            plan_wrapper.get_input_for_component(component_info)
            for component_info in components_dict
        ]

    def pass_input_and_infer(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        shared_tensor_info: SharedTensorInfo,
    ):
        self.input_pool.put_input(component_info, request_info, shared_tensor_info)

        input_list, is_ready = self.input_pool.get_input_if_ready(
            component_info, request_info, self.component_input_dict[component_info]
        )

        if is_ready:
            return self.do_inference(component_info, request_info, input_list)

        return None

    def do_inference(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        shared_input_tensor_info,
    ):
        ## TODO Select a process based on some criteria
        ## TODO When doing this some other thread might have called the inference
        process_idx = request_info.request_idx % len(self.process_dict)
        pipe_wrapper = self.process_dict[process_idx]
        return pipe_wrapper.call_inference(component_info, shared_input_tensor_info)

    def __init_processes(self, processes_per_model, components_dict):
        for i in range(processes_per_model):

            master_conn, worker_conn = multiprocessing.Pipe()
            multiprocessing.Process(
                target=WorkerProcess.start_worker_process,
                args=(components_dict, worker_conn),
            ).start()

            self.process_dict[i] = PipeWrapper(master_conn)
