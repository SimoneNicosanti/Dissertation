import io
import time

import cv2
import grpc
import numpy
import yaml
from PPP.YoloSegmentationPPP import YoloSegmentationPPP
from proto.common_pb2 import ComponentId, ModelId, RequestId
from proto.optimizer_pb2 import OptimizationRequest
from proto.optimizer_pb2_grpc import OptimizationStub
from proto.server_pb2 import InferenceInput, Tensor, TensorChunk, TensorInfo
from proto.server_pb2_grpc import InferenceStub

MAX_CHUNK_SIZE = 3 * 1024 * 1024


def main():
    print("HELLO")
    optimizer_stub: OptimizationStub = OptimizationStub(
        grpc.insecure_channel("optimizer:50060")
    )
    opt_req = OptimizationRequest(
        model_names=["yolo11n-seg"],
        latency_weight=1,
        energy_weight=0,
        device_max_energy=1,
        requests_number=[1],
        deployment_server="0",
    )
    optimizer_stub.serve_optimization(opt_req)

    time.sleep(5)

    server_stub: InferenceStub = InferenceStub(grpc.insecure_channel("localhost:50030"))

    orig_image = cv2.imread("./test/Test_Image.jpg")
    classes = yaml.safe_load(open("./config/coco8.yaml"))["names"]
    yolo_segmentation_ppp = YoloSegmentationPPP(640, 640, classes)

    # Pre-process
    preprocess_dict = yolo_segmentation_ppp.preprocess(orig_image)
    pre_image: numpy.ndarray = preprocess_dict["preprocessed_image"]

    server_stub.do_inference(input_generator(pre_image))


def input_generator(image: numpy.ndarray):

    print("Pre-processed Image")
    tensor_type = image.dtype
    tensor_shape = image.shape

    tensor_bytes = image.tobytes()
    print("Tensor Info {} {} {}".format(tensor_type, tensor_shape, len(tensor_bytes)))
    byte_buffer = io.BytesIO(tensor_bytes)

    component_id = ComponentId(
        model_id=ModelId(model_name="yolo11n-seg", deployer_id="0"),
        server_id="0",
        component_idx="0",
    )
    request_id = RequestId(requester_id="0", request_idx=0)
    tensor_info = TensorInfo(name="images", type=str(tensor_type), shape=tensor_shape)
    while chunk_data := byte_buffer.read(MAX_CHUNK_SIZE):
        tensor_chunk = TensorChunk(chunk_size=len(chunk_data), chunk_data=chunk_data)
        tensor = Tensor(info=tensor_info, tensor_chunk=tensor_chunk)
        print("Sending Chunk")
        yield InferenceInput(
            request_id=request_id,
            component_id=component_id,
            input_tensor=tensor,
        )


if __name__ == "__main__":
    main()
