from proto.RequestConverter import RequestConverter
from proto.server_pb2 import InferenceRequest, InferenceResponse
from proto.server_pb2_grpc import ModelServicer
from RunnerManager import RunnerManager


class ServerClass(ModelServicer):
    def __init__(self, runner_manager: RunnerManager):
        super().__init__()
        self.runner_manager = runner_manager
        pass

    def serve_request(self, request: InferenceRequest, context) -> InferenceResponse:
        request_info, model_input = RequestConverter.convert_input(request)
        output = self.runner_manager.run_request(request_info, model_input)

        if output is not None:
            ## Process and send to next server
            ## Or return back to client
            ## This might be more than one --> Need to know which servers have to receive which tensor
            pass
        else:
            ## Waiting for other inputs to come in order to serve this request
            pass

        return InferenceResponse()
