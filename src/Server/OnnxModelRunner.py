import numpy as np
import onnxruntime as ort
from ModelRunner import ModelRunner


class OnnxModelRunner(ModelRunner):
    def __init__(self, sub_model_path: str):
        super().__init__(sub_model_path)

        ## TODO >> Setup Inference Session
        self.sess = ort.InferenceSession(self.sub_model_path)

        inp: ort.NodeArg
        for inp in self.sess.get_inputs():
            self.input_names.append(inp.name)

        out: ort.NodeArg
        for out in self.onnx_model.graph.output:
            self.output_names.append(out.name)

    def run(self, input_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:

        model_output: list[np.ndarray] = self.sess.run(None, input_feed=input_dict)

        out_dict = {}
        for idx, out_tens in enumerate(model_output):
            out_names = self.sess.get_outputs()
            out_name = out_names[idx]
            out_dict[out_name] = out_tens

        return out_dict
