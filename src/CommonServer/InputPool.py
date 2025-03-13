from CommonServer.InferenceInfo import ComponentInfo, RequestInfo, TensorWrapper


class InputPool:
    def __init__(self):
        self.input_pool: dict[
            tuple[ComponentInfo, RequestInfo], list[TensorWrapper]
        ] = {}

    def put_input(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        tensor_wrapper: TensorWrapper,
    ):

        key = (component_info, request_info)
        self.input_pool.setdefault(key, [])
        self.input_pool[key].append(tensor_wrapper)

    def get_input_if_ready(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        input_list: list[str],
    ) -> list[TensorWrapper] | None:

        key = (component_info, request_info)
        current_inputs = self.input_pool.get(key, None)
        if current_inputs is None:
            return [], False

        current_inputs_names = [
            tensor_wrap.tensor_name for tensor_wrap in current_inputs
        ]

        print("Current Input Names >> ", current_inputs_names)
        print("Input List >> ", input_list)
        if sorted(current_inputs_names) == sorted(input_list):
            return self.input_pool.pop(key), True

        return [], False
