import onnx

from CommonModel.PoolInterface import PoolInterface
from CommonPlan import PlanDecoder
from CommonPlan.Plan import Plan
from CommonPlan.SolvedModelGraph import ComponentId
from ModelDivider.Divide.OnnxModelPartitioner import OnnxModelPartitioner
from proto_compiled.common_pb2 import ComponentId as GrpcComponentId
from proto_compiled.common_pb2 import ModelId
from proto_compiled.model_divide_pb2 import PartitionRequest, PartitionResponse
from proto_compiled.model_divide_pb2_grpc import ModelDivideServicer


class ModelDivideServer(ModelDivideServicer):
    def __init__(self):

        self.pool_interface = PoolInterface()

        pass

    def divide_model(self, request: PartitionRequest, context) -> PartitionResponse:

        model_id: ModelId = request.model_id

        grpc_component_id = GrpcComponentId(
            model_id=model_id, server_id="", component_idx=""
        )
        onnx_model: onnx.ModelProto = self.pool_interface.retrieve_model(
            grpc_component_id
        )

        model_partitioner = OnnxModelPartitioner()

        model_plan: Plan = PlanDecoder.build_plan(
            request.solved_graph_json, model_id.model_name, model_id.deployer_id
        )

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
