import os
from typing import Iterator

import grpc
import onnx

from Common.ConfigReader import ConfigReader
from CommonModel import ModelYielder
from ModelPool.DummyQuantizer import DummyQuantizer
from ModelPool.LayerDivider import LayerDivider
from proto_compiled.common_pb2 import ComponentId
from proto_compiled.model_pool_pb2 import (
    CalibrationChunk,
    CalibrationPullRequest,
    CalibrationPushRequest,
    LayerPullResponse,
    ModelChunk,
    PullRequest,
    PushRequest,
    PushResponse,
)
from proto_compiled.model_pool_pb2_grpc import ModelPoolServicer

MEGABYTE_SIZE = 1024 * 1024


class PoolServer(ModelPoolServicer):
    def __init__(self):
        self.models_dir = ConfigReader("./config/config.ini").read_str(
            "model_pool_dirs", "MODELS_DIR"
        )
        self.components_dir = ConfigReader("./config/config.ini").read_str(
            "model_pool_dirs", "COMPONENTS_DIR"
        )
        self.layers_dir = ConfigReader("./config/config.ini").read_str(
            "model_pool_dirs", "LAYERS_DIR"
        )

        pass

    def push_model(self, request_iterator: Iterator[PushRequest], context):
        print("Push Request")

        first_request: PushRequest = next(request_iterator)
        component_id: ComponentId = first_request.component_id

        model_file_name = self.build_file_name(component_id)
        model_directory = self.build_directory(component_id)
        model_file_path = os.path.join(model_directory, model_file_name)

        print("Writing on File >> ", model_file_name)

        with open(model_file_path, "wb") as model_file:
            model_chunk: ModelChunk = first_request.model_chunk
            model_file.write(model_chunk.chunk_data)

            for push_request in request_iterator:
                model_chunk = push_request.model_chunk
                model_file.write(model_chunk.chunk_data)

        print("Completed Writing on File >> ", model_file_name)

        return PushResponse()

    def pull_model(self, request: PullRequest, context):
        print("Received Pull Request")
        component_id: ComponentId = request.component_id

        model_file_name = self.build_file_name(component_id)
        model_directory = self.build_directory(component_id)
        model_file_path = os.path.join(model_directory, model_file_name)

        print("Asked for file with path >> ", model_file_path)
        try:
            onnx_model: onnx.ModelProto = onnx.load_model(model_file_path)
            yield from ModelYielder.pull_yield(onnx_model)

        except FileNotFoundError:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"File '{model_file_name}' not found.")

            return

    def pull_layer_models(
        self, request: PullRequest, context
    ) -> Iterator[LayerPullResponse]:

        component_id: ComponentId = request.component_id

        ## Not quantized model
        model_file_name = self.build_file_name(component_id)
        model_directory = self.build_directory(component_id)
        model_file_path = os.path.join(model_directory, model_file_name)

        onnx_model = onnx.load_model(model_file_path)
        layer_divider = LayerDivider(model_file_path)

        for layer in onnx_model.graph.node:
            layer_model: onnx.ModelProto = layer_divider.divide_layer(layer.name)
            for pull_response in ModelYielder.pull_yield(layer_model):
                yield LayerPullResponse(
                    layer_name=layer.name,
                    is_quantized=False,
                    model_chunk=pull_response.model_chunk,
                )

        ## Quantized model
        quant_model_file_path = model_file_path.replace(".onnx", "_quant.onnx")

        if not os.path.exists(quant_model_file_path):
            DummyQuantizer.dummy_quantize(model_file_path, quant_model_file_path)

        quantized_layer_divider = LayerDivider(
            model_file_path, quant_onnx_model_path=quant_model_file_path
        )
        for layer in onnx_model.graph.node:
            layer_model: onnx.ModelProto = quantized_layer_divider.divide_layer(
                layer.name
            )
            for pull_response in ModelYielder.pull_yield(layer_model):
                yield LayerPullResponse(
                    layer_name=layer.name,
                    is_quantized=True,
                    model_chunk=pull_response.model_chunk,
                )

        return

    def push_calibration_dataset(
        self, request: Iterator[CalibrationPushRequest], context
    ) -> PushResponse:

        push_request = next(request)
        model_name = push_request.model_id.model_name

        file_name = f"{model_name}_calibration.npy"
        file_dir = ConfigReader("./config/config.ini").read_str(
            "model_pool_dirs", "CALIBRATION_DATASET_DIR"
        )

        file_path = os.path.join(file_dir, file_name)

        with open(file_path, "wb") as file:
            file.write(push_request.calibration_chunk.chunk_data)
            for push_request in request:
                file.write(push_request.calibration_chunk.chunk_data)

        return PushResponse()

    def pull_calibration_dataset(self, request: CalibrationPullRequest, context):

        model_name = request.model_id.model_name

        file_name = f"yolo11_calibration.npy"
        file_dir = ConfigReader().read_str("model_pool_dirs", "CALIBRATION_DATASET_DIR")

        file_path = os.path.join(file_dir, file_name)

        chunk_size = ConfigReader().read_bytes_chunk_size()
        with open(file_path, "rb") as file:
            while chunk_data := file.read(chunk_size):
                print("Here")
                yield CalibrationChunk(chunk_data=chunk_data)

        return

    def build_file_name(self, component_id: ComponentId) -> str:

        model_name: str = component_id.model_id.model_name

        file_name = f"{model_name}"

        server_id: str = component_id.server_id
        model_component_idx: str = component_id.component_idx

        if len(server_id) > 0 and len(model_component_idx) > 0:
            file_name += f"_server_{server_id}_comp_{model_component_idx}"

        return file_name + ".onnx"

    def build_directory(self, component_id: ComponentId, is_layer: bool = False) -> str:

        if is_layer:
            return self.layers_dir

        if len(component_id.server_id) == 0 or len(component_id.component_idx) == 0:
            return self.models_dir
        else:
            return self.components_dir
