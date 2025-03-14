import os

import grpc

from Optimizer.Plan.Plan import Plan
from proto_compiled.common_pb2 import ComponentId, ModelId
from proto_compiled.pool_pb2 import ModelChunk, PushRequest
from proto_compiled.pool_pb2_grpc import ModelPoolStub

POOL_PORT = 50000
MODEL_CHUNK_MAX_SIZE = 3 * 1024 * 1024


class ModelDistributor:

    def __init__(self, divided_model_dir: str):
        self.divided_model_dir = divided_model_dir
        self.pool_connection = grpc.insecure_channel("{}:{}".format("pool", POOL_PORT))

    def distribute(self, model_name, plan: Plan, deployment_server: str):

        for component_id in plan.get_all_components():
            grpc_comp_id = ComponentId(
                model_id=ModelId(model_name=model_name, deployer_id=deployment_server),
                server_id=component_id.net_node_id.node_name,
                component_idx=str(component_id.component_idx),
            )

            if plan.is_component_only_input(component_id):
                continue

            if plan.is_component_only_output(component_id):
                continue

            component_file_path = os.path.join(
                self.divided_model_dir,
                f"{model_name}_depl_{deployment_server}_server_{component_id.net_node_id}_comp_{component_id.component_idx}.onnx",
            )

            pool_stub = ModelPoolStub(self.pool_connection)
            pool_stub.push_model(
                self.__model_chunk_generator(component_file_path, grpc_comp_id)
            )

    def __model_chunk_generator(
        self, component_file_path: str, component_id: ComponentId
    ):
        with open(component_file_path, "rb") as component_file:
            while data_chunk := component_file.read(MODEL_CHUNK_MAX_SIZE):
                model_chunk = ModelChunk(
                    total_chunks=0, chunk_idx=0, chunk_data=data_chunk
                )
                yield PushRequest(component_id=component_id, model_chunk=model_chunk)
