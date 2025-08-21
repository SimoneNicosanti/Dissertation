import io
import time

import numpy as np
import onnx
import onnxruntime
import onnxruntime as ort

from Common import ConfigReader, ProviderInit


class ExecutionProfiler:

    def __init__(self) -> None:
        self.providers = ProviderInit.init_providers_list()
        self.cold_start_times = ConfigReader.ConfigReader().read_int(
            "server_profiler", "COLD_START_RUNS"
        )
        pass

    def profile_exec_time(
        self, onnx_model: onnx.ModelProto, run_times: int, is_quantized: bool
    ) -> dict[bool, float]:

        avg_time, median_time = self.profile_model_exec_time(
            onnx_model, run_times, is_quantized
        )

        return float(avg_time), float(median_time)

    def profile_model_exec_time(
        self, onnx_model: onnx.ModelProto, run_times: int, is_quantized: bool
    ):

        if ProviderInit.test_cuda_ep(self.providers):
            execution_times = self.profile_gpu_execution(
                onnx_model, run_times, is_quantized
            )
        elif ProviderInit.test_openvino_ep(self.providers):
            execution_times = self.profile_openvino_execution(
                onnx_model, run_times, is_quantized
            )
        else:
            execution_times = self.profile_cpu_execution(
                onnx_model, run_times, is_quantized
            )

        execution_times = execution_times * 1e-9

        return (
            np.mean(execution_times),
            np.median(execution_times),
        )

    def profile_gpu_execution(
        self, onnx_model: onnx.ModelProto, run_times: int, is_quantized: bool
    ):
        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=self.providers,
        )

        io_binding = sess.io_binding()

        # Bind input on CUDA
        # We have to move the input on device before starting
        # Otherwise we take into account the transfer time too
        for input in sess.get_inputs():
            elem_type = input.type
            input_elem = np.ones(input.shape, dtype=self.onnx_to_numpy(elem_type))
            ort_input = ort.OrtValue.ortvalue_from_numpy(input_elem, "cuda")
            io_binding.bind_ortvalue_input(input.name, ort_input)

        # Bind output on CUDA
        # We have to preallocate output space on GPU
        # Otherwise the output is moved on CPU automatically
        for output in sess.get_outputs():
            elem_type = output.type
            output_array = np.empty(output.shape, dtype=self.onnx_to_numpy(elem_type))
            ort_output = ort.OrtValue.ortvalue_from_numpy(output_array, "cuda")
            io_binding.bind_ortvalue_output(output.name, ort_output)

        ## Cold Start
        for _ in range(self.cold_start_times):
            sess.run_with_iobinding(io_binding)

        execution_times: np.ndarray = np.zeros(run_times)
        for idx in range(run_times):
            start = time.perf_counter_ns()
            sess.run_with_iobinding(io_binding)
            end = time.perf_counter_ns()

            execution_times[idx] = end - start

        return execution_times

    def profile_openvino_execution(
        self, onnx_model: onnx.ModelProto, run_times: int, is_quantized: bool
    ):
        sess_opt = onnxruntime.SessionOptions()
        # sess_opt.graph_optimization_level = (
        #     onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        # )
        sess = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(),
            providers=self.providers,
            sess_options=sess_opt,
        )
        input_dict = {}
        for elem in sess.get_inputs():
            elem_type = elem.type
            input_elem = np.ones(elem.shape, dtype=self.onnx_to_numpy(elem_type))
            input_dict[elem.name] = ort.OrtValue.ortvalue_from_numpy(input_elem)
        ## Cold Start
        for _ in range(self.cold_start_times):
            sess.run_with_ort_values(None, input_dict)
        execution_times: np.ndarray = np.zeros(run_times)
        for idx in range(run_times):
            start = time.perf_counter_ns()
            sess.run_with_ort_values(None, input_dict)
            end = time.perf_counter_ns()
            execution_times[idx] = end - start
        return execution_times

    def profile_cpu_execution(
        self, onnx_model: onnx.ModelProto, run_times: int, is_quantized: bool
    ):

        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
        )

        input_dict = {}
        for elem in sess.get_inputs():
            elem_type = elem.type
            input_elem = np.ones(elem.shape, dtype=self.onnx_to_numpy(elem_type))
            input_dict[elem.name] = ort.OrtValue.ortvalue_from_numpy(input_elem)

        ## Cold Start
        for _ in range(self.cold_start_times):
            sess.run_with_ort_values(None, input_dict)

        execution_times: np.ndarray = np.zeros(run_times)
        for idx in range(run_times):
            start = time.perf_counter_ns()
            sess.run_with_ort_values(None, input_dict)
            end = time.perf_counter_ns()

            execution_times[idx] = end - start

        return execution_times

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
