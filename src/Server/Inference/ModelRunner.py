import numpy
import onnxruntime as ort

from CommonServer.InferenceInfo import ComponentInfo, TensorWrapper


class ModelRunner:

    def __init__(self, component_dict: dict[ComponentInfo, str]):

        self.component_sessions = {}

        for comp_info, comp_path in component_dict.items():
            comp_session = ort.InferenceSession(comp_path)
            self.component_sessions[comp_info] = comp_session

    def run_component(
        self, component_info: ComponentInfo, input_list: list[TensorWrapper]
    ):

        comp_session = self.component_sessions[component_info]
        comp_out_names = [out.name for out in comp_session.get_outputs()]

        input_dict = {tensor.tensor_name: tensor.numpy_array for tensor in input_list}

        output_list: list[numpy.ndarray] = comp_session.run(
            output_names=comp_out_names,
            input_feed=input_dict,
        )

        return [
            TensorWrapper(
                tensor_name=name,
                tensor_type=str(out.dtype),
                tensor_shape=out.shape,
                numpy_array=out,
            )
            for name, out in zip(comp_out_names, output_list)
        ]
