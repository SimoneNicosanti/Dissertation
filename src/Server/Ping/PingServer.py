from proto_compiled.common_pb2 import Empty
from proto_compiled.ping_pb2_grpc import PingServicer


class PingServer(PingServicer):

    def __init__(self):
        pass

    def latency_test(self, request, context):
        return Empty()

    def bandwidth_test(self, request_stream, context):
        for _ in request_stream:
            pass
        return Empty()
