import threading
from multiprocessing import shared_memory

from CommonServer.InferenceInfo import ModelInfo
from CommonServer.InputReceiver import InputReceiver
from CommonServer.OutputSender import OutputSender
from CommonServer.PlanWrapper import PlanWrapper
from proto_compiled.server_pb2 import InferenceResponse
from proto_compiled.server_pb2_grpc import InferenceServicer


class InferenceServer(InferenceServicer):

    def __init__(self):
        super().__init__()
        self.input_receiver = InputReceiver()
        self.output_sender = OutputSender()

        self.lock = threading.Lock()
        self.plan_wrapper_dict: dict[ModelInfo, PlanWrapper] = {}

    def do_inference(self, input_stream, context):
        ## 1. Receive Input
        component_info, request_info, shared_input_tensor_info = (
            self.input_receiver.handle_input_stream(input_stream)
        )

        ## 2. Do actual inference
        shared_output_tensor_info = self.__do_inference(
            component_info, request_info, shared_input_tensor_info
        )

        if shared_output_tensor_info is not None:
            ## 3. Send Output
            plan_wrapper = self.plan_wrapper_dict[component_info.model_info]
            self.output_sender.send_output(
                plan_wrapper,
                component_info,
                request_info,
                shared_output_tensor_info,
            )

            self.__free_shared_memory(
                shared_input_tensor_info, shared_output_tensor_info
            )

        return InferenceResponse()

    def __do_inference(self, component_info, request_info, shared_input_tensor_info):
        pass

    def __free_shared_memory(self, shared_input_tensor_info, shared_output_tensor_info):
        for input_tensor_info in shared_input_tensor_info:
            sh_mem = shared_memory.SharedMemory(input_tensor_info.shared_memory_name)
            sh_mem.close()
            sh_mem.unlink()

        for output_tensor_info in shared_output_tensor_info:
            sh_mem = shared_memory.SharedMemory(output_tensor_info.shared_memory_name)
            sh_mem.close()
            sh_mem.unlink()
