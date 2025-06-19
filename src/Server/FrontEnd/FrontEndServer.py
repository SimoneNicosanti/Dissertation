import json
import threading
import time
from typing import Generator

from readerwriterlock import rwlock

from CommonIds.ComponentId import ComponentId
from CommonPlan.Plan import Plan
from CommonPlan.WholePlan import WholePlan
from proto_compiled.server_pb2 import AssignmentRequest, AssignmentResponse
from proto_compiled.server_pb2_grpc import InferenceServicer
from Server.FrontEnd import ResponseGenerator
from Server.FrontEnd.ExtremeInferenceManager import ExtremeInferenceManager
from Server.Utils.InferenceInfo import (
    RequestInfo,
    TensorWrapper,
)
from Server.Utils.InferenceManager import InferenceManager
from Server.Utils.InputReceiver import InputReceiver
from Server.Utils.OutputSender import OutputSender

## This has to be a different service running on the client
## Actually it should be started by the deployer --> I.E. The one asking for the plan optimization


class FrontEndServer(InferenceServicer):
    def __init__(self) -> None:
        super().__init__()

        self.input_receiver = InputReceiver()
        self.output_sender = OutputSender()

        self.managers_lock = rwlock.RWLockWriteD()
        self.plan_wrapper_dict: dict[str, Plan] = {}
        self.model_managers: dict[str, InferenceManager] = {}

        self.requests_lock = rwlock.RWLockWriteD()
        self.pending_request_dict: dict[RequestInfo, threading.Event] = {}
        self.final_results_dict: dict[RequestInfo, list[TensorWrapper]] = {}
        pass

    def do_inference(self, input_stream, context):
        ## 1. Receive Input
        component_info, request_info, tensor_wrapper_list = (
            self.input_receiver.handle_input_stream(input_stream)
        )
        print("Received Input")

        ## 2. Do actual inference
        out_tensor_wrap_list = self.__do_inference(
            component_info, request_info, tensor_wrapper_list
        )
        print("Completed Inference")

        if out_tensor_wrap_list is not None:
            ## 3. Send Output
            print("Sending Inference Output")
            with self.managers_lock.gen_rlock():
                plan_wrapper = self.plan_wrapper_dict[component_info.model_name]

            self.output_sender.send_output(
                plan_wrapper,
                component_info,
                request_info,
                out_tensor_wrap_list,
            )
            print("Sent Output")

        ## Lock or Unlock threads for response wait
        ## If the component is the input one --> The thread will be blocked
        ## If the component is the output one --> The input thread will be unlocked
        if out_tensor_wrap_list is not None:
            inference_time = self.lock_unlock_threads(
                request_info, component_info, out_tensor_wrap_list
            )

            ## If Output Component --> Yield output
            if plan_wrapper.is_component_only_input(component_info):
                print("Yielding result")
                with self.requests_lock.gen_wlock():
                    inference_output = self.final_results_dict.pop(request_info)
                yield from ResponseGenerator.yield_response(
                    inference_output, inference_time
                )

        yield

    def lock_unlock_threads(
        self,
        request_info: RequestInfo,
        component_info: ComponentId,
        out_tensor_wrap_list: list[TensorWrapper],
    ) -> float:

        with self.managers_lock.gen_rlock():
            plan_wrapper = self.plan_wrapper_dict[component_info.model_name]

        if plan_wrapper.is_component_only_input(component_info):
            print("Locking Input Thread")

            start_time = time.perf_counter_ns()
            waiting_event = threading.Event()
            with self.requests_lock.gen_wlock():
                self.pending_request_dict[request_info] = waiting_event

            ## We have to lock the thread until the response is received
            waiting_event.wait()
            end_time = time.perf_counter_ns()
            ## Once the thread has been unlocked, we can compute the total
            print(
                f"Time for Request: {request_info.request_idx} >> {(end_time - start_time) * 1e-9} s"
            )

            return (end_time - start_time) * 1e-9

        elif plan_wrapper.is_component_only_output(component_info):
            with self.requests_lock.gen_wlock():
                self.final_results_dict[request_info] = out_tensor_wrap_list
                waiting_event = self.pending_request_dict.pop(request_info)

            print("Unlocking Thread")
            waiting_event.set()

            return 0

    def __do_inference(
        self,
        component_info: ComponentId,
        request_info: RequestInfo,
        tensor_wrap_list: list[TensorWrapper],
    ) -> list[TensorWrapper]:
        print("Do Inference")
        with self.managers_lock.gen_rlock():
            model_manager = self.model_managers[component_info.model_name]
        print("Retrieved Model Manager")
        out_tensor_wrap_list = model_manager.pass_input_and_infer(
            component_info, request_info, tensor_wrap_list
        )
        print("Passed Input to Manager")

        return out_tensor_wrap_list

    def assign_plan(self, assignment_request: AssignmentRequest, context):
        print("Assigning Plan to FrontEnd")
        whole_plan: WholePlan = WholePlan.decode(
            json.loads(assignment_request.optimized_plan)
        )

        for model_name in whole_plan.get_model_names():
            model_plan = whole_plan.get_model_plan(model_name)
            self.register_model(
                model_name,
                model_plan,
            )

        return AssignmentResponse()

    def register_model(self, model_name: str, plan: Plan):

        ## There is no component model, it is just a placeholder
        components_dict = {
            component_id: "" for component_id in plan.get_input_and_output_component()
        }

        with self.managers_lock.gen_wlock():
            self.plan_wrapper_dict[model_name] = plan
            self.model_managers[model_name] = ExtremeInferenceManager(
                plan, components_dict
            )
        pass
