import abc
from multiprocessing import shared_memory

import numpy
from Inference.InferenceInfo import ComponentInfo, RequestInfo, SharedTensorInfo


class ComponentManager(abc.ABC):

    def __init__(
        self,
        component_info: ComponentInfo,
        input_names: list[str],
        output_names: list[str],
    ):

        self.component_info = component_info
        self.input_pool: dict[RequestInfo, SharedTensorInfo] = {}
        self.input_names = input_names
        self.output_names = output_names

    def pass_input_and_infer(
        self, request_info: RequestInfo, shared_tensor_info: SharedTensorInfo
    ) -> dict[str, numpy.ndarray]:
        print("Input Received")
        self.input_pool.setdefault(request_info, [])
        self.input_pool[request_info].append(shared_tensor_info)

        if not self.__is_ready_to_infer(request_info):
            return None

        print("Ready To Infer")
        input_dict, shared_mem_refs = self.prepare_input(request_info)
        inference_result = self.handle_inference_for_request(input_dict)

        self.release_input_memory(request_info, shared_mem_refs)

        return inference_result

    @abc.abstractmethod
    def handle_inference_for_request(
        self, input_dict: dict[str, numpy.ndarray]
    ) -> dict[str, numpy.ndarray]:

        pass

    def release_input_memory(
        self,
        request_info: RequestInfo,
        shared_mem_refs: list[shared_memory.SharedMemory],
    ):
        self.input_pool.pop(request_info)

        for shared_mem_ref in shared_mem_refs:
            shared_mem_ref.close()
            shared_mem_ref.unlink()

    def prepare_input(self, request_info: RequestInfo) -> dict[str, numpy.ndarray]:

        input_dict = {}
        shared_mem_refs = []
        for shared_tensor_info in self.input_pool[request_info]:
            shared_mem_name = shared_tensor_info.shared_memory_name
            shared_mem = shared_memory.SharedMemory(shared_mem_name, create=False)

            input_tensor = numpy.ndarray(
                buffer=shared_mem.buf,
                dtype=shared_tensor_info.tensor_type,
                shape=shared_tensor_info.tensor_shape,
            )

            input_dict[shared_tensor_info.tensor_name] = input_tensor

            shared_mem_refs.append(shared_mem)

        return input_dict, shared_mem_refs

    def __is_ready_to_infer(self, request_info: RequestInfo) -> bool:
        shared_tensor_info_list: SharedTensorInfo = self.input_pool[request_info]
        tensor_names = [
            shared_tensor_info.tensor_name
            for shared_tensor_info in shared_tensor_info_list
        ]

        if sorted(tensor_names) == self.input_names:
            return True

        return
