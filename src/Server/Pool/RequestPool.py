from Model.Model_IO import ModelInput


class RequestKey:
    def __init__(self, model_name: str, client_name: str, requestId: int):
        self.key = (model_name, client_name, requestId)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, value):
        return isinstance(value, RequestKey) and self.key == value.key


class RequestPool:
    def __init__(self):
        self.requests: dict[RequestKey, ModelInput] = {}

    def build_key(
        self,
        model_name: str,
        client_name: str,
        requestId: int,
    ) -> RequestKey:
        return RequestKey(model_name, client_name, requestId)

    def put_request(self, request_key: RequestKey, model_input: ModelInput):
        if request_key in self.requests.keys():
            self.requests.get(request_key).extend_input(model_input)
        else:
            self.requests[request_key] = model_input

    def get_request_input(self, request_key: RequestKey) -> ModelInput:
        model_input = self.requests.pop(request_key)
        return model_input

    def get_request_current_input_names(self, request_key: RequestKey) -> list[str]:
        return self.requests.get(request_key).get_all_input_names()
