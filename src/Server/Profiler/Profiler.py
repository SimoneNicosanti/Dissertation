import json
import os
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort

from CommonQuantization import SoftQuantization


class ExecutionProfiler:

    def __init__(self) -> None:
        pass

    def profile_exec_time(
        self, onnx_model: onnx.ModelProto, run_times: int, is_quantized: bool
    ) -> dict[bool, float]:

        exec_time = self.profile_model_exec_time(onnx_model, run_times, is_quantized)
        # quant_exec_time = self.profile_quant_model_exec_time(onnx_model, run_times)

        return float(exec_time)

    def profile_model_exec_time(
        self, onnx_model: onnx.ModelProto, run_times: int, is_quantized: bool
    ):
        sess_options = ort.SessionOptions()
        sess_options.enable_profiling = True
        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            sess_options=sess_options,
            providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
        )

        input = {}
        for elem in sess.get_inputs():
            elem_type = elem.type
            print(elem_type)
            input[elem.name] = np.zeros(elem.shape, dtype=self.onnx_to_numpy(elem_type))
            # if is_quantized:

            # else:
            #     input[elem.name] = np.zeros(elem.shape, dtype=np.float32)

        ## Cold Stard
        # sess.run(None, input_feed=input)

        # execution_times: np.ndarray = np.zeros(run_times)
        for idx in range(run_times):
            start = time.perf_counter_ns()
            sess.run(None, input_feed=input)
            end = time.perf_counter_ns()

            # execution_times[idx] = end - start

        profile_file_name = sess.end_profiling()
        execution_times: np.ndarray = self.__read_execution_profile(profile_file_name)

        os.remove(profile_file_name)

        return np.mean(execution_times) * 1e-6  ## Returning mean execution time

    # def profile_quant_model_exec_time(
    #     self, onnx_model: onnx.ModelProto, run_times: int
    # ):
    #     temp_file = "hello.onnx"
    #     # temp_file_desc, temp_file = tempfile.mkstemp(suffix=".onnx")
    #     onnx.save_model(onnx_model, temp_file)

    #     prep_model, tensors_range = SoftQuantization.prepare_quantization(
    #         temp_file, ZeroDataReader(onnx_model)
    #     )
    #     quant_model = SoftQuantization.soft_quantization(prep_model, tensors_range)

    #     exec_time = self.profile_normal_model_exec_time(quant_model, run_times)

    #     try:
    #         # os.close(temp_file_desc)
    #         os.remove(temp_file)
    #     except Exception as _:
    #         pass

    #     return exec_time

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

    def onnx_to_numpy(self, onnx_type: str):

        onnx_to_numpy = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(uint8)": np.uint8,
            "tensor(int8)": np.int8,
            "tensor(bool)": np.bool_,
            # Add others as needed
        }

        return onnx_to_numpy[onnx_type]
