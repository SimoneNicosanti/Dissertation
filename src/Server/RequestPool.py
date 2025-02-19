import numpy as np


class RequestKey:
    def __init__(self, model_name: str, client_name: str, requestId: int):
        self.key = (model_name, client_name, requestId)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, value):
        return isinstance(value, RequestKey) and self.key == value.key


class RequestInput:
    def __init__(self):
        self.input_dict: dict[str, np.ndarray] = {}

    def put_input(self, input_name: str, input_value: np.ndarray):
        self.input_dict[input_name] = input_value

    def get_all_inputs(self) -> dict[str, np.ndarray]:
        return self.input_dict

    def get_input_names(self) -> list[str]:
        return list(self.input_dict.keys())


class RequestPool:
    def __init__(self):
        self.requests: dict[RequestKey, RequestInput] = {}

    def build_key(
        self,
        model_name: str,
        client_name: str,
        requestId: int,
    ) -> RequestKey:
        return RequestKey(model_name, client_name, requestId)

    def put_request(
        self,
        request_key: RequestKey,
        input_name: str,
        input_value: np.ndarray,
    ):
        if request_key in self.requests.keys():
            self.requests.get(request_key).put_input(input_name, input_value)
        else:
            request_input = RequestInput()
            request_input.put_input(input_name, input_value)
            self.requests[request_key] = request_input

    def get_request_inputs(self, request_key: RequestKey) -> RequestInput:
        request_input = self.requests.pop(request_key)
        return request_input.get_all_inputs()

    def get_request_current_input_names(self, request_key: RequestKey) -> list[str]:
        return self.requests.get(request_key).get_input_names()
