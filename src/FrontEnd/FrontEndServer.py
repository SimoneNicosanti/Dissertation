import os
import threading
import time

import numpy

from CommonServer.InferenceInfo import (
    ComponentInfo,
    ModelInfo,
    RequestInfo,
    TensorWrapper,
)
from CommonServer.InputReceiver import InputReceiver
from CommonServer.ModelManager import ModelManager
from CommonServer.OutputSender import OutputSender
from CommonServer.PlanWrapper import PlanWrapper
from FrontEnd.ExtremeModelManager import ExtremeModelManager
from proto_compiled.server_pb2 import InferenceResponse
from proto_compiled.server_pb2_grpc import InferenceServicer

## This has to be a different service running on the client
## Actually it should be started by the deployer --> I.E. The one asking for the plan optimization


class FrontEndServer(InferenceServicer):
    def __init__(self, output_save_path: str) -> None:
        super().__init__()

        os.makedirs(output_save_path, exist_ok=True)
        self.output_save_path = output_save_path

        self.input_receiver = InputReceiver()
        self.output_sender = OutputSender()

        self.lock = threading.Lock()
        self.plan_wrapper_dict: dict[ModelInfo, PlanWrapper] = {}

        self.model_managers: dict[ModelInfo, ModelManager] = {}
        self.pending_request_dict: dict[RequestInfo, threading.Event] = {}
        self.pending_times_dict: dict[RequestInfo, int] = {}
        pass

    def do_inference(self, input_stream, context):

        ## 1. Receive Input
        component_info, request_info, input_tensor_info = (
            self.input_receiver.handle_input_stream(input_stream)
        )

        ## 2. Do actual inference
        out_tensor_wrap_list = self.__do_inference(
            component_info, request_info, input_tensor_info
        )

        if out_tensor_wrap_list is not None:
            ## 3. Send Output
            plan_wrapper = self.plan_wrapper_dict[component_info.model_info]
            self.output_sender.send_output(
                plan_wrapper,
                component_info,
                request_info,
                out_tensor_wrap_list,
            )

        ## Lock or Unlock threads for response wait
        if out_tensor_wrap_list is not None:
            if plan_wrapper.is_only_input_component(component_info):
                print("Received Input")
                self.pending_times_dict[request_info] = time.time_ns()
                self.pending_request_dict[request_info] = threading.Event()
                self.pending_request_dict[request_info].wait()
                print(
                    "Total Time >> ",
                    time.time_ns() - self.pending_times_dict[request_info],
                )

            elif plan_wrapper.is_only_output_component(component_info):
                save_tensor_dict = {
                    tensor.tensor_name: tensor.numpy_array
                    for tensor in out_tensor_wrap_list
                }
                out_file_name = "{}_{}.npz".format(
                    request_info.requester_id, request_info.request_idx
                )
                out_path = os.path.join(
                    self.output_save_path,
                    component_info.model_info.model_name,
                    out_file_name,
                )
                numpy.savez(out_path, **save_tensor_dict)
                self.pending_request_dict[request_info].set()

        return InferenceResponse()

    def __do_inference(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        tensor_wrap: TensorWrapper,
    ) -> list[TensorWrapper]:
        model_manager = self.model_managers[component_info.model_info]

        out_tensor_wrap_list = model_manager.pass_input_and_infer(
            component_info, request_info, tensor_wrap
        )

        return out_tensor_wrap_list

    def register_model(
        self,
        model_info: ModelInfo,
        plan_wrapper: PlanWrapper,
        components_dict: dict[ComponentInfo, str],
        processes_per_model: int,
    ):
        model_output_path = os.path.join(self.output_save_path, model_info.model_name)
        os.makedirs(model_output_path, exist_ok=True)

        with self.lock:
            self.plan_wrapper_dict[model_info] = plan_wrapper
            self.model_managers[model_info] = ExtremeModelManager(
                plan_wrapper, components_dict
            )
        pass
