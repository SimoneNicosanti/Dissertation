import tempfile

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_static,
)
from onnxruntime.quantization.preprocess import quant_pre_process


class DummyQuantizer:
    def __init__(self):
        pass

    def dummy_quantize(input_model_path: str, output_model_path: str) -> None:

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as temp_file:
            quant_pre_process(input_model_path, temp_file.name)
            quantize_static(
                temp_file.name,
                output_model_path,
                calibration_data_reader=DummyQuantizer.ZeroDataReader(input_model_path),
            )

    class ZeroDataReader(CalibrationDataReader):

        def __init__(self, model_path: str) -> None:
            sess = ort.InferenceSession(model_path)
            self.input = {}
            for elem in sess.get_inputs():
                self.input[elem.name] = np.zeros(elem.shape, dtype=np.float32)

            self.idx = 0

        def get_next(self):

            if self.idx == 1:
                return None

            self.idx += 1
            return self.input
