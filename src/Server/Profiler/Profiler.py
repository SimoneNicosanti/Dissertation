import json
import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    quant_pre_process,
    quantize_static,
)


class ExecutionProfiler:

    def __init__(self) -> None:
        pass

    def profile_exec_time(
        self, onnx_model: onnx.ModelProto, run_times: int
    ) -> dict[bool, float]:

        exec_time = self.profile_normal_model_exec_time(onnx_model, run_times)
        quant_exec_time = self.profile_quant_model_exec_time(onnx_model, run_times)

        return {False: float(exec_time), True: float(quant_exec_time)}

    def profile_normal_model_exec_time(
        self, onnx_model: onnx.ModelProto, run_times: int
    ):
        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = True
        sess = ort.InferenceSession(
            onnx_model.SerializeToString(), sess_options=sess_options
        )

        input = {}
        for elem in sess.get_inputs():
            input[elem.name] = np.zeros(elem.shape, dtype=np.float32)

        for _ in range(run_times):
            sess.run(None, input_feed=input)

        profile_file_name = sess.end_profiling()
        execution_times: np.ndarray = self.__read_execution_profile(profile_file_name)

        os.remove(profile_file_name)

        return np.mean(execution_times) * 1e-6  ## Returning mean execution time

    def profile_quant_model_exec_time(
        self, onnx_model: onnx.ModelProto, run_times: int
    ):
        _, temp_file = tempfile.mkstemp(suffix=".onnx")
        quant_pre_process(onnx_model, temp_file)
        quantize_static(temp_file, temp_file, ZeroDataReader(onnx_model))
        quantized_model = onnx.load_model(temp_file)

        exec_time = self.profile_normal_model_exec_time(quantized_model, run_times)

        return exec_time

    def __read_execution_profile(self, profile_file_name: str) -> np.ndarray:
        with open(profile_file_name, "r") as profile:
            json_array = json.load(profile)
            filtered_array = filter(
                lambda elem: elem["cat"] == "Session"
                and elem["name"] == "SequentialExecutor::Execute",
                json_array,
            )
            return np.array([elem["dur"] for elem in filtered_array])

        return np.array([])


class ZeroDataReader(CalibrationDataReader):

    def __init__(self, onnx_model: onnx.ModelProto) -> None:
        self.onnx_model = onnx_model

        sess = ort.InferenceSession(onnx_model.SerializeToString())
        self.input = {}
        for elem in sess.get_inputs():
            self.input[elem.name] = np.zeros(elem.shape, dtype=np.float32)

        self.idx = 0

    def get_next(self):

        if self.idx == 1:
            return None

        self.idx += 1
        return self.input
