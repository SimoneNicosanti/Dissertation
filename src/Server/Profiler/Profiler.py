import subprocess
import time

import numpy as np
import onnx
import onnxruntime
import onnxruntime as ort

from Common import ProviderInit


class ExecutionProfiler:

    def __init__(self) -> None:
        self.providers = ProviderInit.init_providers_list()
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

        if ProviderInit.test_cuda_ep(self.providers):
            execution_times = self.profile_gpu_execution(onnx_model, run_times)
        elif ProviderInit.test_openvino_ep(self.providers):
            execution_times = self.profile_openvino_execution(onnx_model, run_times)
        else:
            execution_times = self.profile_cpu_execution(onnx_model, run_times)

        return np.mean(execution_times) * 1e-9  ## Returning mean execution time in sec

    def profile_gpu_execution(self, onnx_model: onnx.ModelProto, run_times: int):
        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=self.providers,
        )

        input_dict = {}
        for elem in sess.get_inputs():
            elem_type = elem.type
            input_elem = np.ones(elem.shape, dtype=self.onnx_to_numpy(elem_type))
            input_dict[elem.name] = ort.OrtValue.ortvalue_from_numpy(input_elem, "cuda")

        ## Cold Start
        for _ in range(3):
            sess.run_with_ort_values(None, input_dict)

        execution_times: np.ndarray = np.zeros(run_times)
        for idx in range(run_times):
            start = time.perf_counter_ns()
            sess.run_with_ort_values(None, input_dict)
            end = time.perf_counter_ns()

            execution_times[idx] = end - start

        return execution_times

    def profile_openvino_execution(self, onnx_model: onnx.ModelProto, run_times: int):

        sess = onnxruntime.InferenceSession(
            onnx_model.SerializeToString(), providers=self.providers
        )

        input_dict = {}
        for elem in sess.get_inputs():
            elem_type = elem.type
            input_elem = np.ones(elem.shape, dtype=self.onnx_to_numpy(elem_type))
            input_dict[elem.name] = ort.OrtValue.ortvalue_from_numpy(input_elem)

        ## Cold Start
        for _ in range(3):
            sess.run_with_ort_values(None, input_dict)

        execution_times: np.ndarray = np.zeros(run_times)
        for idx in range(run_times):
            start = time.perf_counter_ns()
            sess.run_with_ort_values(None, input_dict)
            end = time.perf_counter_ns()

            execution_times[idx] = end - start

        return execution_times

        # from openvino.runtime import Core, Tensor

        # model_bytes = io.BytesIO(onnx_model.SerializeToString())

        # ie = Core()
        # open_vino_model = ie.read_model(model_bytes)
        # compiled_model = ie.compile_model(open_vino_model, "AUTO")

        # sess = ort.InferenceSession(onnx_model.SerializeToString())

        # input_dict = {}
        # for elem in sess.get_inputs():
        #     elem_type = elem.type
        #     input_dict[elem.name] = np.ones(
        #         elem.shape, dtype=self.onnx_to_numpy(elem_type)
        #     )
        # del sess

        # for input_name in input_dict.keys():
        #     input_dict[input_name] = Tensor(array=input_dict[input_name])

        # ## Cold Start
        # for _ in range(3):
        #     compiled_model.infer_new_request(input_dict)

        # execution_times: np.ndarray = np.zeros(run_times)
        # for idx in range(run_times):
        #     start = time.perf_counter_ns()
        #     compiled_model.infer_new_request(input_dict)
        #     end = time.perf_counter_ns()

        #     execution_times[idx] = end - start

        # return execution_times

    def profile_cpu_execution(self, onnx_model: onnx.ModelProto, run_times: int):

        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
        )

        input_dict = {}
        for elem in sess.get_inputs():
            elem_type = elem.type
            input_elem = np.ones(elem.shape, dtype=self.onnx_to_numpy(elem_type))
            input_dict[elem.name] = ort.OrtValue.ortvalue_from_numpy(input_elem)

        ## Cold Start
        for _ in range(3):
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
