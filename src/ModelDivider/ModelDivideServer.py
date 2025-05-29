import json

import onnx

from CommonModel.PoolInterface import PoolInterface
from CommonPlan.Plan import Plan
from CommonPlan.SolvedModelGraph import ComponentId
from CommonPlan.WholePlan import WholePlan
from ModelDivider.Divide.OnnxModelPartitioner import OnnxModelPartitioner
from proto_compiled.common_pb2 import ComponentId as GrpcComponentId
from proto_compiled.common_pb2 import ModelId
from proto_compiled.model_divide_pb2 import PartitionRequest, PartitionResponse
from proto_compiled.model_divide_pb2_grpc import ModelDivideServicer


class ModelDivideServer(ModelDivideServicer):
    def __init__(self):

        self.pool_interface = PoolInterface()

        pass

    def divide_model(
        self, partition_request: PartitionRequest, context
    ) -> PartitionResponse:

        whole_plan: WholePlan = WholePlan.decode(
            json.loads(partition_request.solved_graph_json)
        )

        for model_name in whole_plan.get_model_names():
            model_id: ModelId = ModelId(model_name=model_name, deployer_id="")

            grpc_component_id = GrpcComponentId(
                model_id=model_id, server_id="", component_idx=""
            )
            onnx_model: onnx.ModelProto = self.pool_interface.retrieve_model(
                grpc_component_id
            )

            model_partitioner = OnnxModelPartitioner()

            model_plan: Plan = whole_plan.get_model_plan(model_name)

            divided_components: dict[ComponentId, onnx.ModelProto] = (
                model_partitioner.partition_model(model_plan, onnx_model)
            )

            for component_id, model in divided_components.items():
                grpc_component_id = GrpcComponentId(
                    model_id=model_id,
                    server_id=component_id.net_node_id.node_name,
                    component_idx=str(component_id.component_idx),
                )
                self.pool_interface.save_model(grpc_component_id, model)

        return PartitionResponse()
