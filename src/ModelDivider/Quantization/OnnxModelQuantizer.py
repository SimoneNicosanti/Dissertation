import tempfile

import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.preprocess import quant_pre_process
from onnxruntime.quantization.quantize import quantize_static


class OnnxModelQuantizer:

    @staticmethod
    def quantize_model(
        onnx_model: onnx.ModelProto,
        calibration_dataset: np.ndarray,
        quantized_layers: list[str],
    ) -> onnx.ModelProto:
        print("Quantizing Model")
        _, temp_file_name = tempfile.mkstemp(suffix=".onnx")

        quant_pre_process(onnx_model, temp_file_name)

        _, temp_file_name_quant = tempfile.mkstemp(suffix=".onnx")
        quantize_static(
            temp_file_name,
            temp_file_name_quant,
            calibration_data_reader=OnnxModelQuantizer.OnnxDataReader(
                temp_file_name, calibration_dataset
            ),
            nodes_to_quantize=quantized_layers,
            extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
        )

        quant_onnx_model = onnx.load_model(temp_file_name_quant)

        # for node in quant_onnx_model.graph.node:
        #     if (
        #         "/model.23/proto/upsample/ConvTranspose_output_0_QuantizeLinear_Output"
        #         in node.output
        #     ):
        #         print("Exists")

        onnx.save_model(quant_onnx_model, "/model_pool_data/components/final.onnx")

        return quant_onnx_model

        pass

    class OnnxDataReader(CalibrationDataReader):
        def __init__(self, model_path: str, calibration_set: np.ndarray):
            sess = onnxruntime.InferenceSession(model_path)
            input_info = sess.get_inputs()
            del sess

            self.input_names = [input.name for input in input_info]

            self.curr_elem = 0
            self.calibration_set = calibration_set

            pass

        def get_next(self):

            if self.curr_elem >= len(self.calibration_set):
                return None

            input_dict = {}
            for input_name in self.input_names:
                input_elem = self.calibration_set[self.curr_elem]
                input_dict[input_name] = np.expand_dims(input_elem, axis=0)

            self.curr_elem += 1
            return input_dict
