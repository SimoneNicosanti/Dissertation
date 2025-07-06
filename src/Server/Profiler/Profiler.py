import json
import time

import numpy as np
import onnx
import onnxruntime as ort


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
        # sess_options.enable_profiling = True
        # sess_options.graph_optimization_level = (
        #     ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # )
        # sess_options.graph_optimization_level = (
        #     ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # )

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        elif "OpenVINOExecutionProvider" in ort.get_available_providers():
            providers.append("OpenVINOExecutionProvider")
        else:
            providers.append("CPUExecutionProvider")

        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            sess_options=sess_options,
            providers=providers,
        )

        input_dict = {}
        for elem in sess.get_inputs():
            elem_type = elem.type
            input_elem = np.ones(elem.shape, dtype=self.onnx_to_numpy(elem_type))
            if "CUDAExecutionProvider" in ort.get_available_providers():
                ## Moving data to GPU only once to reduce data transfer impact on profiling
                input_elem = ort.OrtValue.ortvalue_from_numpy(
                    input_elem, "cuda"
                )

            input_dict[elem.name] = input_elem

        ## Cold Stard --> Loading model with first inference
        if "CUDAExecutionProvider" in ort.get_available_providers():
            sess.run_with_ort_values(None, input_feed=input_dict)
        else:
            sess.run(None, input_feed=input_dict)

        execution_times: np.ndarray = np.zeros(run_times)
        for idx in range(run_times):
            start = time.perf_counter_ns()

            if "CUDAExecutionProvider" in ort.get_available_providers():
                ## In this way the output remains on the GPU
                ## We do not have the impact of data transfer
                sess.run_with_ort_values(None, input_feed=input_dict)
            else:
                sess.run(None, input_feed=input_dict)
            
            end = time.perf_counter_ns()

            execution_times[idx] = end - start

        # profile_file_name = sess.end_profiling()
        # execution_times: np.ndarray = self.__read_execution_profile(profile_file_name)

        # os.remove(profile_file_name)

        return np.mean(execution_times) * 1e-9  ## Returning mean execution time in sec


    def __read_execution_profile(self, profile_file_name: str) -> np.ndarray:
        with open(profile_file_name, "r") as profile:
            json_array = json.load(profile)
            print(json_array)
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
