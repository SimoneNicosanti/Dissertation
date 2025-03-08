from multiprocessing import shared_memory

import numpy
import onnxruntime as ort
from Inference.InputInfo import ComponentInfo, RequestInfo, SharedTensorInfo
from Inference.QueueWrapper import QueueWrapper
from Wrapper.PlanWrapper import PlanWrapper


class ComponentManager:

    def __init__(
        self,
        component_info: ComponentInfo,
        component_path: str,
        is_only_input: bool,
        is_only_output: bool,
    ):
        self.component_info = component_info
        self.input_pool: dict[RequestInfo, SharedTensorInfo] = {}

        self.is_only_input = is_only_input
        self.is_only_output = is_only_output

        if not (is_only_input or is_only_output):
            ## Init inference session for this component
            self.inference_session: ort.InferenceSession = ort.InferenceSession(
                component_path
            )
            self.component_input_names = []
            self.component_output_names = []
        else:
            ## Maybe you can take info from the plan since you have it
            pass

    def pass_input(
        self, request_info: RequestInfo, shared_tensor_info: SharedTensorInfo
    ) -> dict[str, numpy.ndarray]:

        self.input_pool.setdefault(request_info, [])
        self.input_pool[request_info].append(shared_tensor_info)

        if not self.__is_ready_to_infer(request_info):
            return None

        return self.handle_inference_for_request(request_info)

        pass

    def handle_inference_for_request(
        self, request_info: RequestInfo
    ) -> dict[str, numpy.ndarray]:

        input_dict = self.prepare_input(request_info)

        if not (self.is_only_input or self.is_only_output):
            outputs_list = self.inference_session.run(
                output_names=self.component_output_names, input_feed=input_dict
            )

            output_dict = dict(zip(self.component_output_names, outputs_list))
        else:
            output_dict = {}
            for input_name in input_dict.keys():
                output_dict[input_name] = numpy.copy(input_dict[input_name])

        self.release_input_memory(request_info)

        return output_dict

    def release_input_memory(self, request_info: RequestInfo):
        for shared_tensor_info in self.input_pool[request_info]:
            shared_mem_name = shared_tensor_info.shared_memory_name
            shared_mem = shared_memory.SharedMemory(shared_mem_name)
            shared_mem.close()
            shared_mem.unlink()

    def prepare_input(self, request_info: RequestInfo) -> dict[str, numpy.ndarray]:

        input_dict = {}

        for shared_tensor_info in self.input_pool[request_info]:
            shared_mem_name = shared_tensor_info.shared_memory_name
            shared_mem = shared_memory.SharedMemory(shared_mem_name)

            input_tensor = numpy.frombuffer(
                buffer=shared_mem.buf, dtype=shared_tensor_info.tensor_type
            ).reshape(shared_tensor_info.tensor_shape)

            input_dict[shared_tensor_info.tensor_name] = input_tensor

        return input_dict

    def __is_ready_to_infer(self, request_info: RequestInfo) -> bool:
        shared_tensor_info_list: SharedTensorInfo = self.input_pool[request_info]
        tensor_names = [
            shared_tensor_info.tensor_name
            for shared_tensor_info in shared_tensor_info_list
        ]

        if tensor_names == self.component_input_names:
            return True

        return


def work(queue_wrapper: QueueWrapper, plan_wrapper: PlanWrapper) -> None:

    component_manager_dict = {}

    while True:

        extracted_data: tuple = queue_wrapper.extract_from_queue()
        if extracted_data[0] == QueueWrapper.SPAWN_MESSAGE:
            component_info = extracted_data[1]
            component_path = extracted_data[2]
            handle_component_spawn(
                component_info, component_path, component_manager_dict, plan_wrapper
            )

        elif extracted_data[0] == QueueWrapper.INPUT_MESSAGE:
            component_info = extracted_data[1]
            request_info = extracted_data[2]
            shared_tensor_info = extracted_data[3]
            handle_input_pass(
                component_info, request_info, shared_tensor_info, component_manager_dict
            )
        else:
            raise Exception("Unknown queue")


def handle_component_spawn(
    component_info: ComponentInfo,
    component_path: str,
    component_manager_dict: dict[ComponentInfo, ComponentManager],
    plan_wrapper: PlanWrapper,
):
    is_only_input = plan_wrapper.is_only_input_component(component_info)
    is_only_output = plan_wrapper.is_only_output_component(component_info)
    print("Handling Spawn for component {}".format(component_path))
    if component_info not in component_manager_dict:
        component_manager_dict[component_info] = ComponentManager(
            component_info, component_path, is_only_input, is_only_output
        )


def handle_input_pass(
    component_info: ComponentInfo,
    request_info: RequestInfo,
    shared_tensor_info: SharedTensorInfo,
    component_manager_dict: dict[ComponentInfo, ComponentManager],
):

    component_manager = component_manager_dict[component_info]

    infer_output = component_manager.pass_input(request_info, shared_tensor_info)
    if infer_output is None:
        ## Inference not ready for this request
        return

    ## TODO Handle inference output
    ## Follow plan and send to next components
