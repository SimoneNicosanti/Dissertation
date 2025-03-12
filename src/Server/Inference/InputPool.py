import threading

from CommonServer.InferenceInfo import ComponentInfo, RequestInfo, SharedTensorInfo


class InputPool:
    def __init__(self):
        self.lock = threading.Lock()
        self.input_pool: dict[
            tuple[ComponentInfo, RequestInfo], list[SharedTensorInfo]
        ] = {}

    def put_input(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        shared_tensor_info: SharedTensorInfo,
    ):
        with self.lock:
            key = (component_info, request_info)
            self.input_pool.setdefault(key, [])
            self.input_pool[key].append(shared_tensor_info)

    def get_input_if_ready(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        input_list: list[str],
    ) -> list[SharedTensorInfo] | None:
        with self.lock:
            key = (component_info, request_info)
            current_inputs = self.input_pool.get(key, None)
            if current_inputs is None:
                return [], False

            current_inputs_names = [
                shared_tensor_info.tensor_name for shared_tensor_info in current_inputs
            ]

            if sorted(current_inputs_names) == sorted(input_list):
                return self.input_pool.pop(key), True

        return [], False
