import numpy as np
from ModelRunner import ModelRunner
from RequestPool import RequestKey, RequestPool


class RunnerManager:

    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.request_pool = RequestPool()

    def run_request(
        self,
        model_name: str,
        client_name: str,
        requestId: int,
        input_name: str,
        input_value: np.ndarray,
    ) -> dict[str, np.ndarray] | None:

        request_key = self.request_pool.build_key(model_name, client_name, requestId)
        self.request_pool.put_request(request_key, input_name, input_value)

        if self.__is_inference_ready(request_key):
            return self.model_runner.run(
                self.request_pool.get_request_inputs(request_key)
            )
        return None

    def __is_inference_ready(self, request_key: RequestKey) -> bool:
        return (
            self.model_runner.get_input_names()
            == self.request_pool.get_request_current_input_names(request_key)
        )
