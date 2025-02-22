import numpy as np
from Model.Model_IO import ModelInput, ModelOutput
from proto.server_pb2 import Data, InferenceRequest, Info
from RunnerManager import RequestInfo


class RequestConverter:
    def __init__(self):
        pass

    @staticmethod
    def convert_input(request: InferenceRequest) -> tuple[RequestInfo, ModelInput]:
        info: Info = request.info
        request_info = RequestInfo(info.model_name, info.client_id, info.request_id)

        model_input = ModelInput()
        for input in request.inputs:
            input_name = input.name
            input_value = np.frombuffer(input.data, dtype=np.dtype(input.type)).reshape(
                input.shape
            )

            model_input.add_input(input_name=input_name, input_value=input_value)

        return request_info, model_input

    @staticmethod
    def convert_output(
        request_info: RequestInfo, model_output: ModelOutput
    ) -> InferenceRequest:
        info = Info(
            request_id=request_info.request_id,
            client_id=request_info.client_name,
            model_name=request_info.model_name,
        )

        next_inp_list = []
        for out_name, out_value in model_output.get_all_outputs().items():
            new_inp = Data(
                name=out_name,
                type=str(out_value.dtype),
                shape=out_value.shape,
                data=out_value.tobytes(),
            )

            next_inp_list.append(new_inp)

        return InferenceRequest(info=info, inputs=next_inp_list)
