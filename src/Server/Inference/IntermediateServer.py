import asyncio
import threading

from CommonServer.InferenceInfo import (
    ComponentInfo,
    ModelInfo,
    RequestInfo,
    TensorWrapper,
)
from CommonServer.InputReceiver import InputReceiver
from CommonServer.OutputSender import OutputSender
from CommonServer.PlanWrapper import PlanWrapper
from proto_compiled.server_pb2 import InferenceResponse
from proto_compiled.server_pb2_grpc import InferenceServicer
from Server.Inference.IntermediateModelManager import IntermediateModelManager


class IntermediateServer(InferenceServicer):
    def __init__(self):
        super().__init__()

        self.input_receiver = InputReceiver()
        self.output_sender = OutputSender()

        self.lock = threading.Lock()
        self.plan_wrapper_dict: dict[ModelInfo, PlanWrapper] = {}

        self.model_managers: dict[ModelInfo, IntermediateModelManager] = {}

    def do_inference(self, input_stream, context):
        ## 1. Receive Input
        component_info, request_info, tensor_wrapper_list = (
            self.input_receiver.handle_input_stream(input_stream)
        )
        print("Completed Input Reading")

        ## 2. Do actual inference
        ## Running in async way allows not to block the sender thread
        threading.Thread(
            target=self.__do_inference_and_send_output,
            args=(component_info, request_info, tensor_wrapper_list),
        ).start()

        ## Returning an empty stream
        ## For the intermediate part of the inference we are not intrested in the output value
        yield

    def __do_inference_and_send_output(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        tensor_wrapper_list: TensorWrapper,
    ):
        model_manager = self.model_managers[component_info.model_info]

        output_tensor_info = model_manager.pass_input_and_infer(
            component_info, request_info, tensor_wrapper_list
        )

        if output_tensor_info is not None:
            ## 3. Send Output
            plan_wrapper = self.plan_wrapper_dict[component_info.model_info]
            self.output_sender.send_output(
                plan_wrapper,
                component_info,
                request_info,
                output_tensor_info,
            )

    def register_model(
        self,
        model_info: ModelInfo,
        plan_wrapper: PlanWrapper,
        components_dict: dict[ComponentInfo, str],
        threads_per_model: int,
    ):

        with self.lock:
            if model_info not in self.model_managers:
                self.model_managers[model_info] = IntermediateModelManager(
                    plan_wrapper, components_dict, threads_per_model
                )
            if model_info not in self.plan_wrapper_dict:
                self.plan_wrapper_dict[model_info] = plan_wrapper
