import numpy
from Inference.ComponentManager import ComponentManager
from Inference.ComponentManagerInput import ComponentManagerInput
from Inference.ComponentManagerOutput import ComponentManagerOutput
from Inference.InferenceInfo import ComponentInfo, RequestInfo, SharedTensorInfo
from Inference.OutputSender import OutputSender
from Wrapper.PlanWrapper import PlanWrapper
from Wrapper.QueueWrapper import QueueWrapper


class WorkerProcess:
    def __init__(self, queue_wrapper: QueueWrapper, plan_wrapper: PlanWrapper):
        self.queue_wrapper = queue_wrapper
        self.plan_wrapper = plan_wrapper
        self.component_manager_dict = {}

        self.output_sender = OutputSender(self.plan_wrapper)

    def do_work(self):
        while True:

            extracted_data: tuple = self.queue_wrapper.extract_from_queue()
            if extracted_data[0] == QueueWrapper.SPAWN_MESSAGE:
                component_info = extracted_data[1]
                component_path = extracted_data[2]
                self.handle_component_spawn(component_info, component_path)

            elif extracted_data[0] == QueueWrapper.INPUT_MESSAGE:
                component_info = extracted_data[1]
                request_info = extracted_data[2]
                shared_tensor_info = extracted_data[3]
                self.handle_input_pass(
                    component_info,
                    request_info,
                    shared_tensor_info,
                )
            else:
                raise Exception("Unknown queue")

    def handle_component_spawn(
        self,
        component_info: ComponentInfo,
        component_path: str,
    ):
        is_only_input = self.plan_wrapper.is_only_input_component(component_info)
        is_only_output = self.plan_wrapper.is_only_output_component(component_info)
        print("Handling Spawn for component {}".format(component_path))
        if component_info in self.component_manager_dict:
            return

        if is_only_input:
            pass
        elif is_only_output:
            pass
        else:
            self.component_manager_dict[component_info] = ComponentManager(
                component_info, component_path, is_only_input, is_only_output
            )

    def handle_input_pass(
        self,
        component_info: ComponentInfo,
        request_info: RequestInfo,
        shared_tensor_info: SharedTensorInfo,
    ):

        component_manager: ComponentManager = self.component_manager_dict[
            component_info
        ]

        infer_output: dict[str, numpy.ndarray] = component_manager.pass_input_and_infer(
            request_info, shared_tensor_info
        )
        if infer_output is None:
            ## Inference not ready for this request
            return

        self.output_sender.send_output(component_info, request_info, infer_output)


def work(queue_wrapper: QueueWrapper, plan_wrapper: PlanWrapper) -> None:
    WorkerProcess(queue_wrapper, plan_wrapper).do_work()
