from Server.Inference.InputReceiver import InputReceiver
from proto.server_pb2_grpc import InferenceServicer
from Server.Inference.RunnerManager import RunnerManager


class ServerClass(InferenceServicer):
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
