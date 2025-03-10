import io

import grpc
import numpy
from Inference.InferenceInfo import ComponentInfo, RequestInfo
from proto.common_pb2 import ComponentId, ModelId, RequestId
from proto.register_pb2 import ReachabilityInfo, ServerId
from proto.register_pb2_grpc import RegisterStub
from proto.server_pb2 import InferenceInput, Tensor, TensorChunk, TensorInfo
from proto.server_pb2_grpc import InferenceStub
from Wrapper.PlanWrapper import PlanWrapper

MAX_CHUNK_SIZE = 3 * 1024 * 1024


class OutputSender:
    def __init__(self, plan_wrapper: PlanWrapper):
        self.plan_wrapper = plan_wrapper

        self.register_stub = RegisterStub(grpc.insecure_channel("registry:50051"))

        self.next_server_stubs: dict[str, InferenceStub] = {}
        pass

    def send_output(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        infer_output: dict[str, numpy.ndarray],
    ):
        print("Sending Inference Output")
        next_components_dict: dict[str, list[ComponentInfo]] = (
            self.plan_wrapper.find_next_connections(component_info)
        )

        for tensor_name in infer_output.keys():
            print("Sending  {}".format(tensor_name))
            for next_comp_info in next_components_dict[tensor_name]:
                next_comp_stub: InferenceStub = (
                    self.__open_connection_to_next_component(next_comp_info)
                )

                next_comp_stub.do_inference(
                    self.__stream_generator(
                        next_comp_info,
                        request_info,
                        infer_output[tensor_name],
                        tensor_name,
                    )
                )

        pass

    def __stream_generator(
        self,
        next_component_info: ComponentInfo,
        request_info: RequestInfo,
        input_tensor: numpy.ndarray,
        input_tensor_name: str,
    ):

        # print(
        #     "Tensor Info {} {} {}".format(tensor_type, tensor_shape, len(tensor_bytes))
        # )

        component_id = ComponentId(
            model_id=ModelId(
                model_name=next_component_info.model_info.model_name,
                deployer_id=next_component_info.model_info.deployer_id,
            ),
            server_id=next_component_info.server_id,
            component_idx=next_component_info.component_idx,
        )
        request_id = RequestId(
            requester_id=request_info.requester_id, request_idx=request_info.request_idx
        )

        tensor_type = input_tensor.dtype
        tensor_shape = input_tensor.shape

        tensor_info = TensorInfo(
            name=input_tensor_name, type=str(tensor_type), shape=tensor_shape
        )

        tensor_bytes = input_tensor.tobytes()
        byte_buffer = io.BytesIO(tensor_bytes)
        while chunk_data := byte_buffer.read(MAX_CHUNK_SIZE):
            tensor_chunk = TensorChunk(
                chunk_size=len(chunk_data), chunk_data=chunk_data
            )
            tensor = Tensor(info=tensor_info, tensor_chunk=tensor_chunk)

            yield InferenceInput(
                request_id=request_id,
                component_id=component_id,
                input_tensor=tensor,
            )

    def __open_connection_to_next_component(self, next_component_info: ComponentInfo):

        if next_component_info.server_id not in self.next_server_stubs.keys():

            reachability_info: ReachabilityInfo = self.register_stub.get_info_from_id(
                ServerId(server_id=next_component_info.server_id),
            )

            server_stub = InferenceStub(
                grpc.insecure_channel(
                    f"{reachability_info.ip_address}:{reachability_info.inference_port}"
                )
            )

            self.next_server_stubs[next_component_info.server_id] = server_stub

        return self.next_server_stubs[next_component_info.server_id]
