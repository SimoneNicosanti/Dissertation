from CommonIds.ComponentId import ComponentId
from Server.Utils.InferenceInfo import RequestInfo, TensorWrapper


class InputPool:
    def __init__(self):
        self.input_pool: dict[tuple[ComponentId, RequestInfo], list[TensorWrapper]] = {}

    def put_input(
        self,
        component_id: ComponentId,
        request_info: RequestInfo,
        tensor_wrapper: TensorWrapper,
    ):
        key = (component_id, request_info)
        print(key)
        self.input_pool.setdefault(key, [])
        self.input_pool[key].append(tensor_wrapper)

    def get_input_if_ready(
        self,
        component_id: ComponentId,
        request_info: RequestInfo,
        input_list: list[str],
    ) -> list[TensorWrapper]:
        key = (component_id, request_info)
        print(key)

        current_inputs = self.input_pool.get(key, None)
        if current_inputs is None:
            return [], False
        print("Got Inputs from Dict")
        current_inputs_names = [
            tensor_wrap.tensor_name for tensor_wrap in current_inputs
        ]

        if sorted(current_inputs_names) == sorted(input_list):
            return self.input_pool.pop(key), True

        return [], False
