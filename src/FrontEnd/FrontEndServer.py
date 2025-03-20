import threading
import time
from typing import Generator

from readerwriterlock import rwlock

from CommonServer.InferenceInfo import (
    ComponentInfo,
    ModelInfo,
    RequestInfo,
    TensorWrapper,
)
from CommonServer.InputReceiver import InputReceiver
from CommonServer.InferenceManager import InferenceManager
from CommonServer.OutputSender import OutputSender
from CommonServer.PlanWrapper import PlanWrapper
from FrontEnd import ResponseGenerator
from FrontEnd.ExtremeInferenceManager import ExtremeInferenceManager
from proto_compiled.server_pb2_grpc import InferenceServicer

## This has to be a different service running on the client
## Actually it should be started by the deployer --> I.E. The one asking for the plan optimization


class FrontEndServer(InferenceServicer):
    def __init__(self) -> None:
        super().__init__()

        self.input_receiver = InputReceiver()
        self.output_sender = OutputSender()

        self.managers_lock = rwlock.RWLockWriteD()
        self.plan_wrapper_dict: dict[ModelInfo, PlanWrapper] = {}
        self.model_managers: dict[ModelInfo, InferenceManager] = {}

        self.requests_lock = rwlock.RWLockWriteD()
        self.pending_request_dict: dict[RequestInfo, threading.Event] = {}
        self.final_results_dict: dict[RequestInfo, list[TensorWrapper]] = {}
        pass

    def do_inference(self, input_stream, context):

        ## 1. Receive Input
        component_info, request_info, tensor_wrapper_list = (
            self.input_receiver.handle_input_stream(input_stream)
        )

        ## 2. Do actual inference
        out_tensor_wrap_list = self.__do_inference(
            component_info, request_info, tensor_wrapper_list
        )

        if out_tensor_wrap_list is not None:
            ## 3. Send Output
            print("Sending Inference Output")
            with self.managers_lock.gen_rlock():
                plan_wrapper = self.plan_wrapper_dict[component_info.model_info]

            self.output_sender.send_output(
                plan_wrapper,
                component_info,
                request_info,
                out_tensor_wrap_list,
            )

        ## Lock or Unlock threads for response wait
        ## If the component is the input one --> The thread will be blocked
        ## If the component is the output one --> The input thread will be unlocked
        if out_tensor_wrap_list is not None:
            self.lock_unlock_threads(request_info, component_info, out_tensor_wrap_list)

            ## If Input Component --> Yield output
            if plan_wrapper.is_only_input_component(component_info):
                print("Yielding result")
                with self.requests_lock.gen_wlock():
                    inference_output = self.final_results_dict.pop(request_info)
                yield from ResponseGenerator.yield_response(inference_output)

        yield

    def lock_unlock_threads(
        self,
        request_info: RequestInfo,
        component_info: ComponentInfo,
        out_tensor_wrap_list: list[TensorWrapper],
    ) -> Generator:

        with self.managers_lock.gen_rlock():
            plan_wrapper = self.plan_wrapper_dict[component_info.model_info]

        if plan_wrapper.is_only_input_component(component_info):
            print("Locking Input Thread")

            start_time = time.time_ns()
            waiting_event = threading.Event()
            with self.requests_lock.gen_wlock():
                self.pending_request_dict[request_info] = waiting_event

            ## We have to lock the thread until the response is received
            waiting_event.wait()
            end_time = time.time_ns()
            ## Once the thread has been unlocked, we can compute the total
            print(
                f"Time for Request: {request_info.request_idx} >> {(end_time - start_time) / 1e9} s"
            )

        elif plan_wrapper.is_only_output_component(component_info):
            with self.requests_lock.gen_wlock():
                self.final_results_dict[request_info] = out_tensor_wrap_list
                waiting_event = self.pending_request_dict.pop(request_info)

            print("Unlocking Thread")
            waiting_event.set()

    def __do_inference(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        tensor_wrap_list: list[TensorWrapper],
    ) -> list[TensorWrapper]:

        with self.managers_lock.gen_rlock():
            model_manager = self.model_managers[component_info.model_info]

        out_tensor_wrap_list = model_manager.pass_input_and_infer(
            component_info, request_info, tensor_wrap_list
        )

        return out_tensor_wrap_list

    def register_model(
        self,
        model_info: ModelInfo,
        plan_wrapper: PlanWrapper,
        components_dict: dict[ComponentInfo, str],
        threads_per_model: int,
    ):

        with self.managers_lock.gen_wlock():
            self.plan_wrapper_dict[model_info] = plan_wrapper

            self.model_managers[model_info] = ExtremeInferenceManager(
                plan_wrapper, components_dict
            )
        pass
