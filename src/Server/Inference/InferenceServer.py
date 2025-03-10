from Inference.InputReceiver import InputReceiver
from Inference.ModelManagerPool import ModelManagerPool
from proto.server_pb2 import InferenceResponse
from proto.server_pb2_grpc import InferenceServicer


class InferenceServer(InferenceServicer):
    def __init__(self, model_manager_pool: ModelManagerPool):
        super().__init__()
        self.model_manager_pool = model_manager_pool
        pass

    def do_inference(self, request_iterator, context):
        print("Received Inference Request")
        input_receiver = InputReceiver()
        print("Creating Input Receiver")
        component_info, request_info, shared_tensor_info = (
            input_receiver.handle_input_stream(request_iterator)
        )

        self.model_manager_pool.pass_input(
            component_info, request_info, shared_tensor_info
        )

        return InferenceResponse()
