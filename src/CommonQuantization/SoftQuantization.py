import copy
import os
import tempfile
from pathlib import Path

import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
)
import onnxruntime
from onnxruntime.quantization.calibrate import TensorsData, create_calibrator
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.quant_utils import load_model_with_shape_infer
from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry

def prepare_quantization(
    model_path: str, calibration_data_reader: CalibrationDataReader
) -> onnx.ModelProto:

    augmented_model_path = model_path.replace(".onnx", "_augmented.onnx")

    providers = []
    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        providers.append("CUDAExecutionProvider")
    elif "OpenVINOExecutionProvider" in onnxruntime.get_available_providers():
        providers.append("OpenVINOExecutionProvider")
    else:
        providers.append("CPUExecutionProvider")

    calibrator = create_calibrator(
        model_path,
        augmented_model_path=augmented_model_path,
        use_external_data_format=False,
        extra_options={},  ## TODO. In our case we are using no calibration option
        providers=providers,
    )
    calibrator.collect_data(calibration_data_reader)
    tensors_range = calibrator.compute_data()

    model = load_model_with_shape_infer(Path(model_path))

    try:
        os.remove(augmented_model_path)
        # os.close(aug_file_desc)
    except Exception as e:
        pass

    return model, tensors_range


def soft_quantization(
    model: onnx.ModelProto,
    tensors_range: TensorsData,
    nodes_to_quantize: list[str] = None,
    nodes_to_exclude: list[str] = None,
    weight_qType: QuantType = QuantType.QInt8,
    activation_qType: QuantType = QuantType.QInt8,
    per_channel: bool = False,
    reduce_range: bool = False,
):
    nodes_to_quantize = [] if nodes_to_quantize is None else nodes_to_quantize
    nodes_to_exclude = [] if nodes_to_exclude is None else nodes_to_exclude

    op_types_to_quantize = list(
        set(list(QLinearOpsRegistry.keys()) + list(QDQRegistry.keys()))
    )

    model_copy = copy.deepcopy(model)

    quantized_model = QDQQuantizer(
        model=model_copy,
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_qType=weight_qType,
        activation_qType=activation_qType,
        tensors_range=tensors_range,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        op_types_to_quantize=op_types_to_quantize,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    ).quantize_model()

    return quantized_model
