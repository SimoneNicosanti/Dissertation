import os

import grpc
from Plan.Plan import Plan
from proto.common_pb2 import ModelBlockId
from proto.pool_pb2 import ModelChunk, PushRequest
from proto.pool_pb2_grpc import ModelPoolStub

POOL_PORT = 50000
MODEL_CHUNK_MAX_SIZE = 3 *  1024 * 1024


class ModelDistributor:

    def __init__(self, divided_model_dir: str):
        self.divided_model_dir = divided_model_dir
        self.pool_stub = ModelPoolStub(
            grpc.insecure_channel("{}:{}".format("pool", POOL_PORT))
        )

    def distribute(self, model_name, plan: Plan):
        print("Distributing model {}".format(model_name))

        for component_id in plan.get_all_components():
            print("Pushing component {}".format(component_id))
            model_block_id = ModelBlockId(
                model_name=model_name,
                server_id=component_id.net_node_id.node_name,
                block_idx=str(component_id.component_idx),
            )
            print("Here 1")

            component_file_path = os.path.join(
                self.divided_model_dir,
                f"{model_name}_server_{component_id.net_node_id}_comp_{component_id.component_idx}.onnx",
            )

            print(component_file_path)

            self.pool_stub.push_model(
                self.__model_chunk_generator(component_file_path, model_block_id)
            )

    def __model_chunk_generator(
        self, component_file_path: str, model_block_id: ModelBlockId
    ):
        with open(component_file_path, "rb") as component_file:
            while data_chunk := component_file.read(MODEL_CHUNK_MAX_SIZE):
                print("Sending Chunk of size {}".format(len(data_chunk)))
                model_chunk = ModelChunk(
                    total_chunks=0, chunk_idx=0, chunk_data=data_chunk
                )
                yield PushRequest(
                    model_block_id=model_block_id, model_chunk=model_chunk
                )
