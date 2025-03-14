import io

import grpc
from readerwriterlock import rwlock

from CommonServer.InferenceInfo import ComponentInfo, RequestInfo, TensorWrapper
from CommonServer.PlanWrapper import PlanWrapper
from proto_compiled.common_pb2 import ComponentId, ModelId, RequestId
from proto_compiled.register_pb2 import ReachabilityInfo, ServerId
from proto_compiled.register_pb2_grpc import RegisterStub
from proto_compiled.server_pb2 import InferenceInput, Tensor, TensorChunk, TensorInfo
from proto_compiled.server_pb2_grpc import InferenceStub

MAX_CHUNK_SIZE = 3 * 1024 * 1024


class OutputSender:
    def __init__(self):
        self.registry_connection = grpc.insecure_channel("registry:50051")

        self.server_channel_lock = rwlock.RWLockWriteD()
        self.next_server_channel: dict[str, grpc.Channel] = {}

        self.callback_channel_lock = rwlock.RWLockWriteD()
        self.callback_channel_dict: dict[str, grpc.Channel] = {}

        self.reachability_info_lock = rwlock.RWLockWriteD()
        self.reachability_info_dict: dict[str, ReachabilityInfo] = {}

    def send_output(
        self,
        plan_wrapper: PlanWrapper,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        infer_output: list[TensorWrapper],
    ):
        next_components_dict: dict[ComponentInfo, list[str]] = (
            plan_wrapper.find_next_connections(component_info)
        )

        for next_comp_info in next_components_dict.keys():
            next_comp_inputs = next_components_dict[next_comp_info]

            to_send_list = filter(
                lambda item: item.tensor_name in next_comp_inputs, infer_output
            )

            next_comp_stub: InferenceStub = self.__get_stub_for_next_server(
                plan_wrapper, next_comp_info, request_info
            )

            response_stream = next_comp_stub.do_inference(
                self.__stream_generator(next_comp_info, request_info, to_send_list)
            )

            ## As the receiver will answer with a stream
            ## We have to consume it in order to unlock the computation
            ## The stream will actually be empty, but we have to do it anyway
            for _ in response_stream:
                print("BELLAAAH")
                pass

        pass

    def __stream_generator(
        self,
        next_component_info: ComponentInfo,
        request_info: RequestInfo,
        to_send_list: list[TensorWrapper],
    ):

        component_id = ComponentId(
            model_id=ModelId(
                model_name=next_component_info.model_info.model_name,
                deployer_id=next_component_info.model_info.deployer_id,
            ),
            server_id=next_component_info.server_id,
            component_idx=next_component_info.component_idx,
        )
        request_id = RequestId(
            requester_id=request_info.requester_id,
            request_idx=request_info.request_idx,
            callback_port=request_info.callback_port,
        )

        for tensor_wrapper in to_send_list:
            tensor_info = TensorInfo(
                name=tensor_wrapper.tensor_name,
                type=tensor_wrapper.tensor_type,
                shape=tensor_wrapper.tensor_shape,
            )

            byte_buffer = io.BytesIO(tensor_wrapper.numpy_array.tobytes())
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

    def __get_stub_for_next_server(
        self,
        plan_wrapper: PlanWrapper,
        next_component_info: ComponentInfo,
        request_info: RequestInfo,
    ):

        with self.server_channel_lock.gen_rlock():
            is_present = (
                next_component_info.server_id in self.reachability_info_dict.keys()
            )

            if is_present:
                reachability_info = self.reachability_info_dict[
                    next_component_info.server_id
                ]

        if not is_present:
            reachability_info: ReachabilityInfo = RegisterStub(
                self.registry_connection
            ).get_info_from_id(
                ServerId(server_id=next_component_info.server_id),
            )
            self.reachability_info_dict[next_component_info.server_id] = (
                reachability_info
            )

        if plan_wrapper.is_only_output_component(next_component_info):

            with self.callback_channel_lock.gen_rlock():
                is_present = (
                    next_component_info.server_id in self.callback_channel_dict.keys()
                )

                if is_present:
                    channel = self.callback_channel_dict[next_component_info.server_id]

            if not is_present:
                with self.callback_channel_lock.gen_wlock():
                    ## Chek if it has not been changed in the meantime
                    if (
                        next_component_info.server_id
                        in self.callback_channel_dict.keys()
                    ):
                        channel = self.callback_channel_dict[
                            next_component_info.server_id
                        ]

                    else:
                        channel = grpc.insecure_channel(
                            f"{reachability_info.ip_address}:{request_info.callback_port}"
                        )

                        self.callback_channel_dict[next_component_info.server_id] = (
                            channel
                        )

        else:
            with self.server_channel_lock.gen_rlock():
                is_present = (
                    next_component_info.server_id in self.next_server_channel.keys()
                )

                if is_present:
                    channel = self.next_server_channel[next_component_info.server_id]

            if not is_present:
                with self.server_channel_lock.gen_wlock():
                    ## Chek if it has not been changed in the meantime
                    if next_component_info.server_id in self.next_server_channel.keys():
                        channel = self.next_server_channel[
                            next_component_info.server_id
                        ]

                    else:
                        channel = grpc.insecure_channel(
                            f"{reachability_info.ip_address}:{reachability_info.inference_port}"
                        )

                        self.next_server_channel[next_component_info.server_id] = (
                            channel
                        )

        server_stub = InferenceStub(channel)

        return server_stub
