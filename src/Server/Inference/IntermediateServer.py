import gc
import json
import threading
import time

import grpc
from readerwriterlock import rwlock

from Common.ConfigReader import ConfigReader
from CommonIds.ComponentId import ComponentId
from CommonPlan.Plan import Plan
from CommonPlan.WholePlan import WholePlan
from proto_compiled.server_pb2 import AssignmentRequest, AssignmentResponse
from proto_compiled.server_pb2_grpc import InferenceServicer, InferenceStub
from Server.Fetcher.Fetcher import Fetcher
from Server.Inference.IntermediateInferenceManager import IntermediateInferenceManager
from Server.Utils.InferenceInfo import (
    RequestInfo,
    TensorWrapper,
)
from Server.Utils.InputReceiver import InputReceiver
from Server.Utils.OutputSender import OutputSender


class IntermediateServer(InferenceServicer):
    def __init__(self, server_id: str):
        super().__init__()

        self.input_receiver = InputReceiver()
        self.output_sender = OutputSender()

        self.fetcher = Fetcher(server_id)
        self.server_id: str = server_id

        self.lock = rwlock.RWLockWriteD()
        self.plan_dict: dict[str, Plan] = {}
        self.model_managers: dict[str, IntermediateInferenceManager] = {}

    def do_inference(self, input_stream, context):
        ## 1. Receive Input
        component_info, request_info, tensor_wrapper_list = (
            self.input_receiver.handle_input_stream(input_stream)
        )

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
        component_id: ComponentId,
        request_info: RequestInfo,
        tensor_wrapper_list: list[TensorWrapper],
    ):
        with self.lock.gen_rlock():
            model_manager = self.model_managers[component_id.model_name]

        output_tensor_info = model_manager.pass_input_and_infer(
            component_id, request_info, tensor_wrapper_list
        )

        if output_tensor_info is not None:
            ## 3. Send Output
            with self.lock.gen_rlock():
                plan_wrapper = self.plan_dict[component_id.model_name]

            self.output_sender.send_output(
                plan_wrapper,
                component_id,
                request_info,
                output_tensor_info,
            )

    def assign_plan(self, assignment_request: AssignmentRequest, context):

        whole_plan: WholePlan = WholePlan.decode(
            json.loads(assignment_request.optimized_plan)
        )

        for model_name in whole_plan.get_model_names():
            model_plan = whole_plan.get_model_plan(model_name)
            self.register_model(
                model_name,
                model_plan,
            )

        ## If this is the start server, then we send the plan to the local FrontEnd Component
        if whole_plan.get_start_server() == self.server_id:
            addr = "localhost"
            port = ConfigReader().read_int("ports", "FRONTEND_PORT")
            conn = grpc.insecure_channel("{}:{}".format(addr, port))
            frontend_server = InferenceStub(conn)
            frontend_server.assign_plan(assignment_request)

        return AssignmentResponse()

    def register_model(
        self,
        model_name: str,
        plan: Plan,
    ):
        ## Fetch model
        ## Fetch components
        components_dict = self.fetcher.fetch_components(plan)

        threads_per_model = 1

        with self.lock.gen_wlock():
            prev_manager = self.model_managers.pop(model_name, None)
            if prev_manager is not None:
                del prev_manager
                gc.collect()

            self.model_managers[model_name] = IntermediateInferenceManager(
                plan, components_dict, threads_per_model
            )
            self.plan_dict[model_name] = plan
