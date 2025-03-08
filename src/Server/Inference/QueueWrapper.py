import multiprocessing

from Inference.InputInfo import ComponentInfo, ModelInfo, RequestInfo, SharedTensorInfo

MAX_QUEUE_SIZE = 1_000


class QueueWrapper:

    INPUT_MESSAGE = 0
    SPAWN_MESSAGE = 1

    def __init__(self):
        self.comm_queue = multiprocessing.Queue(MAX_QUEUE_SIZE)

    def order_component_spawn(self, component_info: ComponentInfo, component_path: str):
        to_send = (QueueWrapper.SPAWN_MESSAGE, component_info, component_path)
        self.comm_queue.put(to_send)

    def pass_input(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        shared_tensor_info: SharedTensorInfo,
    ):
        to_send = (
            QueueWrapper.INPUT_MESSAGE,
            component_info,
            request_info,
            shared_tensor_info,
        )
        self.comm_queue.put(to_send)

    def get_all_queues(self):
        return self.input_queue, self.spawn_queue

    def extract_from_queue(self):
        extracted_value = self.comm_queue.get()
        return extracted_value
