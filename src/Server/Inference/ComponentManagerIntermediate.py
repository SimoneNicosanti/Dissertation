import numpy
import onnxruntime as ort
from Inference.ComponentManager import ComponentManager
from InferenceInfo import ComponentInfo


class ComponentManagerIntermediate(ComponentManager):

    def __init__(
        self,
        component_info: ComponentInfo,
        input_names: list[str],
        output_names: list[str],
        component_path: str,
    ):
        super().__init__(component_info, input_names, output_names)

        self.inference_session = ort.InferenceSession(component_path)
        self.inferred_output_names = [
            out.name for out in self.inference_session.get_outputs()
        ]

    def handle_inference_for_request(
        self, input_dict: dict[str, numpy.ndarray]
    ) -> dict[str, numpy.ndarray]:
        output_dict = {}
        outputs_list = self.inference_session.run(
            output_names=self.component_output_names, input_feed=input_dict
        )

        output_dict = dict(zip(self.inferred_output_names, outputs_list))

        print("Inference Done")
        return output_dict
