from InputReceiver import InputReceiver
from proto.inference_pb2 import ModelInput, SendInputResponse
from proto.inference_pb2_grpc import ModelServicer
from RunnerManager import RunnerManager


class ServerClass(ModelServicer):
    def __init__(self, runner_manager: RunnerManager):
        super().__init__()
        self.runner_manager = runner_manager
        pass

    def send_input(self, request_iterator, context):
        input_receiver = InputReceiver()
        input: ModelInput
        for input in request_iterator:
            input_receiver.handle_input(input)

        input_receiver.get_input()
