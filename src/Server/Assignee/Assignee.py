import json
import os

import onnx

from Common import ConfigReader
from CommonIds.ComponentId import ComponentId
from CommonModel.PoolInterface import PoolInterface
from CommonPlan.Plan import Plan
from CommonPlan.WholePlan import WholePlan
from proto_compiled.common_pb2 import ComponentId as GrpcComponentId
from proto_compiled.common_pb2 import ModelId
from proto_compiled.server_pb2 import AssignmentRequest, AssignmentResponse
from proto_compiled.server_pb2_grpc import AssigneeServicer
from Server.Inference.IntermediateServer import IntermediateServer


class Fetcher(AssigneeServicer):

    def __init__(
        self,
        server_id: str,
        intermediate_server: IntermediateServer,
    ):
        self.server_id: str = server_id

        local_model_dir = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "server_dirs", "MODELS_DIR"
        )
        self.local_model_dir: str = local_model_dir

        self.intermediate_server: IntermediateServer = intermediate_server

        self.model_pool_interface = PoolInterface()

    def send_plan(self, assignment_request: AssignmentRequest, context):
        print("Received Plan")

        whole_plan: WholePlan = WholePlan.decode(
            json.loads(assignment_request.optimized_plan)
        )

        for model_name in whole_plan.get_model_names():
            model_plan = whole_plan.get_model_plan(model_name)
            self.__handle_model_plan(model_plan)

        return AssignmentResponse()

    def __handle_model_plan(self, plan: Plan):

        assigned_components: list[ComponentId] = plan.get_server_components(
            self.server_id
        )
        paths_dict = {}
        model_info = plan.get_model_info()

        for comp_id in assigned_components:

            if plan.is_component_only_input(comp_id) or plan.is_component_only_output(
                comp_id
            ):
                ## These components will be handled by the front end
                continue
            else:
                component_path = self.__fetch_component(
                    comp_id,
                )

                paths_dict[comp_id] = component_path

        if len(paths_dict) != 0:
            print("Registring Model")
            self.intermediate_server.register_model(model_info, plan, paths_dict, 10)

    def __fetch_component(self, component_id: ComponentId):

        model_name = component_id.model_name
        server_id = component_id.net_node_id.node_name
        component_idx = component_id.component_idx

        grpc_component_id = GrpcComponentId(
            model_id=ModelId(model_name=model_name),
            server_id=server_id,
            component_idx=str(component_idx),
        )
        print("Handle Model Plan")
        onnx_model: onnx.ModelProto = self.model_pool_interface.retrieve_model(
            grpc_component_id
        )
        print("Saved Component")
        component_path = self.build_component_path(component_id)
        onnx.save_model(onnx_model, component_path)

        return component_path

    def build_component_path(self, component_id: ComponentId):
        return os.path.join(
            self.local_model_dir,
            "{}_server_{}_comp_{}.onnx".format(
                component_id.model_name,
                component_id.net_node_id.node_name,
                component_id.component_idx,
            ),
        )
