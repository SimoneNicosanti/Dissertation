import os

import onnx

from Common import ConfigReader
from CommonIds.ComponentId import ComponentId
from CommonModel.PoolInterface import PoolInterface
from CommonPlan.Plan import Plan
from proto_compiled.common_pb2 import ComponentId as GrpcComponentId
from proto_compiled.common_pb2 import ModelId


class Fetcher:

    def __init__(
        self,
        server_id: str,
    ):
        self.server_id: str = server_id

        local_model_dir = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "server_dirs", "MODELS_DIR"
        )
        self.local_model_dir: str = local_model_dir

        self.model_pool_interface = PoolInterface()

    def fetch_components(self, plan: Plan) -> dict[ComponentId, str]:

        assigned_components: list[ComponentId] = plan.get_server_components(
            self.server_id
        )
        paths_dict = {}

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

        return paths_dict

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
