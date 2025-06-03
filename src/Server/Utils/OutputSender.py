import io

import grpc
from readerwriterlock import rwlock

from Common import ConfigReader
from CommonIds.ComponentId import ComponentId
from CommonIds.NodeId import NodeId
from CommonPlan.Plan import Plan
from proto_compiled.common_pb2 import ComponentId as GrpcComponentId
from proto_compiled.common_pb2 import ModelId, RequestId
from proto_compiled.register_pb2 import ReachabilityInfo, ServerId
from proto_compiled.register_pb2_grpc import RegisterStub
from proto_compiled.server_pb2 import InferenceInput, Tensor, TensorChunk, TensorInfo
from proto_compiled.server_pb2_grpc import InferenceStub
from Server.Utils.InferenceInfo import RequestInfo, TensorWrapper


class OutputSender:
    def __init__(self):
        registry_addr = ConfigReader.ConfigReader().read_str(
            "addresses", "REGISTRY_ADDR"
        )
        registry_port = ConfigReader.ConfigReader().read_int("ports", "REGISTRY_PORT")
        self.registry_connection = grpc.insecure_channel(
            "{}:{}".format(registry_addr, registry_port)
        )

        self.chunk_size_bytes = ConfigReader.ConfigReader().read_bytes_chunk_size()

        self.server_channel_lock = rwlock.RWLockWriteD()
        self.next_server_channel: dict[NodeId, grpc.Channel] = {}

        self.callback_channel_lock = rwlock.RWLockWriteD()
        self.callback_channel_dict: dict[NodeId, grpc.Channel] = {}

        self.reachability_info_lock = rwlock.RWLockWriteD()
        self.reachability_info_dict: dict[NodeId, ReachabilityInfo] = {}

    def send_output(
        self,
        plan: Plan,
        component_id: ComponentId,
        request_info: RequestInfo,
        infer_output: list[TensorWrapper],
    ):
        next_components_dict: dict[ComponentId, list[str]] = plan.find_next_connections(
            component_id
        )

        for next_comp_info in next_components_dict.keys():
            next_comp_inputs = next_components_dict[next_comp_info]

            to_send_list = filter(
                lambda item: item.tensor_name in next_comp_inputs, infer_output
            )

            next_comp_stub: InferenceStub = self.__get_stub_for_next_server(
                plan, next_comp_info, request_info
            )

            response_stream = next_comp_stub.do_inference(
                self.__stream_generator(next_comp_info, request_info, to_send_list)
            )

            ## As the receiver will answer with a stream
            ## We have to consume it in order to unlock the computation
            ## The stream will actually be empty, but we have to do it anyway
            for _ in response_stream:
                pass

        pass

    def __stream_generator(
        self,
        next_component_id: ComponentId,
        request_info: RequestInfo,
        to_send_list: list[TensorWrapper],
    ):

        component_id = GrpcComponentId(
            model_id=ModelId(
                model_name=next_component_id.model_name,
            ),
            server_id=next_component_id.net_node_id.node_name,
            component_idx=str(next_component_id.component_idx),
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
            while chunk_data := byte_buffer.read(self.chunk_size_bytes):
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
        plan: Plan,
        next_component_id: ComponentId,
        request_info: RequestInfo,
    ):

        with self.server_channel_lock.gen_rlock():
            is_present = (
                next_component_id.net_node_id in self.reachability_info_dict.keys()
            )

            if is_present:
                reachability_info = self.reachability_info_dict[
                    next_component_id.net_node_id
                ]

        if not is_present:
            reachability_info: ReachabilityInfo = RegisterStub(
                self.registry_connection
            ).get_info_from_id(
                ServerId(server_id=next_component_id.net_node_id.node_name),
            )
            self.reachability_info_dict[next_component_id.net_node_id] = (
                reachability_info
            )

        if plan.is_component_only_output(next_component_id):

            with self.callback_channel_lock.gen_rlock():
                is_present = (
                    next_component_id.net_node_id in self.callback_channel_dict.keys()
                )

                if is_present:
                    channel = self.callback_channel_dict[next_component_id.net_node_id]

            if not is_present:
                with self.callback_channel_lock.gen_wlock():
                    ## Chek if it has not been changed in the meantime
                    if (
                        next_component_id.net_node_id
                        in self.callback_channel_dict.keys()
                    ):
                        channel = self.callback_channel_dict[
                            next_component_id.net_node_id
                        ]

                    else:
                        channel = grpc.insecure_channel(
                            f"{reachability_info.ip_address}:{request_info.callback_port}"
                        )

                        self.callback_channel_dict[next_component_id.net_node_id] = (
                            channel
                        )

        else:
            with self.server_channel_lock.gen_rlock():
                is_present = (
                    next_component_id.net_node_id in self.next_server_channel.keys()
                )

                if is_present:
                    channel = self.next_server_channel[next_component_id.net_node_id]

            if not is_present:
                with self.server_channel_lock.gen_wlock():
                    ## Chek if it has not been changed in the meantime
                    if next_component_id.net_node_id in self.next_server_channel.keys():
                        channel = self.next_server_channel[
                            next_component_id.net_node_id
                        ]

                    else:
                        channel = grpc.insecure_channel(
                            f"{reachability_info.ip_address}:{reachability_info.inference_port}"
                        )

                        self.next_server_channel[next_component_id.net_node_id] = (
                            channel
                        )

        server_stub = InferenceStub(channel)

        return server_stub
