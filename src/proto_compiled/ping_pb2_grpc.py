# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import common_pb2 as common__pb2

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
        + f' but the generated code in ping_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class PingStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.latency_test = channel.unary_unary(
                '/optimizer.Ping/latency_test',
                request_serializer=common__pb2.Empty.SerializeToString,
                response_deserializer=common__pb2.Empty.FromString,
                _registered_method=True)


class PingServicer(object):
    """Missing associated documentation comment in .proto file."""

    def latency_test(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_PingServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'latency_test': grpc.unary_unary_rpc_method_handler(
                    servicer.latency_test,
                    request_deserializer=common__pb2.Empty.FromString,
                    response_serializer=common__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'optimizer.Ping', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('optimizer.Ping', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class Ping(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def latency_test(request,
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
            '/optimizer.Ping/latency_test',
            common__pb2.Empty.SerializeToString,
            common__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
