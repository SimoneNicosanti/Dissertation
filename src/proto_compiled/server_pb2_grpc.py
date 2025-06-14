# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import server_pb2 as server__pb2

GRPC_GENERATED_VERSION = '1.71.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in server_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class InferenceStub(object):
    """Inference
    Inference service exposed by the server in order to do distributed inference
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.do_inference = channel.stream_stream(
                '/server.Inference/do_inference',
                request_serializer=server__pb2.InferenceInput.SerializeToString,
                response_deserializer=server__pb2.InferenceResponse.FromString,
                _registered_method=True)
        self.assign_plan = channel.unary_unary(
                '/server.Inference/assign_plan',
                request_serializer=server__pb2.AssignmentRequest.SerializeToString,
                response_deserializer=server__pb2.AssignmentResponse.FromString,
                _registered_method=True)


class InferenceServicer(object):
    """Inference
    Inference service exposed by the server in order to do distributed inference
    """

    def do_inference(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def assign_plan(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InferenceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'do_inference': grpc.stream_stream_rpc_method_handler(
                    servicer.do_inference,
                    request_deserializer=server__pb2.InferenceInput.FromString,
                    response_serializer=server__pb2.InferenceResponse.SerializeToString,
            ),
            'assign_plan': grpc.unary_unary_rpc_method_handler(
                    servicer.assign_plan,
                    request_deserializer=server__pb2.AssignmentRequest.FromString,
                    response_serializer=server__pb2.AssignmentResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'server.Inference', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('server.Inference', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class Inference(object):
    """Inference
    Inference service exposed by the server in order to do distributed inference
    """

    @staticmethod
    def do_inference(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            '/server.Inference/do_inference',
            server__pb2.InferenceInput.SerializeToString,
            server__pb2.InferenceResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def assign_plan(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/server.Inference/assign_plan',
            server__pb2.AssignmentRequest.SerializeToString,
            server__pb2.AssignmentResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)


class ExecutionProfileStub(object):
    """Execution Profile
    Service exposed in order to build execution profile for a certain model
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.profile_execution = channel.unary_unary(
                '/server.ExecutionProfile/profile_execution',
                request_serializer=server__pb2.ExecutionProfileRequest.SerializeToString,
                response_deserializer=server__pb2.ExecutionProfileResponse.FromString,
                _registered_method=True)


class ExecutionProfileServicer(object):
    """Execution Profile
    Service exposed in order to build execution profile for a certain model
    """

    def profile_execution(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ExecutionProfileServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'profile_execution': grpc.unary_unary_rpc_method_handler(
                    servicer.profile_execution,
                    request_deserializer=server__pb2.ExecutionProfileRequest.FromString,
                    response_serializer=server__pb2.ExecutionProfileResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'server.ExecutionProfile', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('server.ExecutionProfile', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class ExecutionProfile(object):
    """Execution Profile
    Service exposed in order to build execution profile for a certain model
    """

    @staticmethod
    def profile_execution(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/server.ExecutionProfile/profile_execution',
            server__pb2.ExecutionProfileRequest.SerializeToString,
            server__pb2.ExecutionProfileResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
