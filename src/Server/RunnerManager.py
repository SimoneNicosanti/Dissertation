from dataclasses import dataclass

from Model.Model_IO import ModelInput, ModelOutput
from Model.ModelRunner import ModelRunner
from Pool.RequestPool import RequestKey, RequestPool


@dataclass
class RequestInfo:
    def __init__(self, model_name: str, client_name: str, request_id: int):
        self.model_name = model_name
        self.client_name = client_name
        self.request_id = request_id


class RunnerManager:

    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.request_pool = RequestPool()

    def run_request(
        self,
        request_info: RequestInfo,
        model_input: ModelInput,
    ) -> ModelOutput | None:

        request_key = self.request_pool.build_key(
            request_info.model_name, request_info.client_name, request_info.requestId
        )
        self.request_pool.put_request(request_key, model_input)

        if self.__is_inference_ready(request_key):
            model_input: ModelInput = self.request_pool.get_request_input(request_key)
            return self.model_runner.run(model_input)

        return None

    def __is_inference_ready(self, request_key: RequestKey) -> bool:
        return (
            self.model_runner.get_input_names()
            == self.request_pool.get_request_current_input_names(request_key)
        )
