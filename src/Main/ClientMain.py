import io
import time

import cv2
import grpc
import numpy
import yaml

from Client.PPP.YoloSegmentationPPP import YoloSegmentationPPP
from proto_compiled.common_pb2 import ComponentId, ModelId, RequestId
from proto_compiled.optimizer_pb2 import OptimizationRequest
from proto_compiled.optimizer_pb2_grpc import OptimizationStub
from proto_compiled.server_pb2 import InferenceInput, Tensor, TensorChunk, TensorInfo
from proto_compiled.server_pb2_grpc import InferenceStub

MAX_CHUNK_SIZE = 3 * 1024 * 1024


def main():

    time.sleep(5)

    server_stub: InferenceStub = InferenceStub(grpc.insecure_channel("localhost:50090"))

    classes = yaml.safe_load(open("./Client/config/coco8.yaml"))["names"]
    yolo_segmentation_ppp = YoloSegmentationPPP(640, 640, classes)

    # Pre-process
    orig_image = cv2.imread("./Client/test/Test_Image.jpg")
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
