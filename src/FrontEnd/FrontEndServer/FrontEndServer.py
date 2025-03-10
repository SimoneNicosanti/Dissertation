import cv2
import grpc
import numpy
import yaml
from proto.common_pb2 import Empty, ModelId, RequestId
from proto.frontend_pb2 import CallbackInfo, InferenceInfo, InferenceReturn
from proto.frontend_pb2_grpc import (
    FrontEndServicer,
)
from proto.server_pb2_grpc import InferenceStub

from FrontEnd.PPP.YoloSegmentationPPP import YoloSegmentationPPP


class FrontEndServer(FrontEndServicer):

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = output_dir
        self.request_idx = 0
        self.pending_requests: list[RequestId] = []

        classes = yaml.safe_load(open("./config/coco8.yaml"))["names"]
        self.yolo_ppp = YoloSegmentationPPP(640, 640, classes)

        self.inference_stub: InferenceStub = InferenceStub(
            grpc.insecure_channel("localhost:50030")
        )

    def start_inference(self, inference_info: InferenceInfo, context):

        model_id = inference_info.model_id
        model_name = model_id.model_name

        for input_path in inference_info.input_files_paths:
            input_image = cv2.imread(input_path)

            preprocess_dict = self.yolo_ppp.preprocess(input_image)
            prep_image = preprocess_dict["preprocessed_image"]

            ## Call this asinchronously
            self.inference_stub.do_inference(self.input_generator(prep_image))

        pass

    def result_callback(self, callback_info: CallbackInfo, context):
        pass

    def input_generator(self, image: numpy.ndarray):
        pass
