import tempfile
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
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
                quant_format=QuantFormat.QDQ,
                extra_options={
                    "ActivationSymmetric": True,
                    "WeightSymmetric": True,
                    "QuantizeBias": False,
                },
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


# Percorso del modello
model_path = "/model_pool_data/models/yolo11x-seg.onnx"
quant_model_path = model_path.replace(".onnx", "_quant_1.onnx")

DummyQuantizer.dummy_quantize(model_path, quant_model_path)

# Configura il sessione ONNX Runtime con CUDA Execution Provider
providers = [
    (
        "TensorrtExecutionProvider",
        {},
    )
]
# Fallback a CPU se la GPU non è disponibile
if "CUDAExecutionProvider" not in ort.get_available_providers():
    print("CUDAExecutionProvider non disponibile, uso CPU.")
    providers = ["CPUExecutionProvider"]

# Crea la sessione
sess_options = ort.SessionOptions()
session = ort.InferenceSession(
    quant_model_path, sess_options=sess_options, providers=providers
)

# Ottieni input e crea dummy input
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_dtype = np.float32  # di solito FP32 per modelli ONNX standard

# Rimpiazza eventuali dimensioni dinamiche (None) con 1
input_shape = [s if isinstance(s, int) else 1 for s in input_shape]
dummy_input = np.random.rand(*input_shape).astype(input_dtype)

# Esegui 50 inferenze
print("Eseguo Cold Start...")
for _ in range(5):
    session.run(None, {input_name: dummy_input})

# Esegui 50 inferenze
print("Eseguo il modello per 50 iterazioni...")
start = time.perf_counter_ns()
for _ in range(50):
    session.run(None, {input_name: dummy_input})
end = time.perf_counter_ns()

# Stampa risultati
total_time = (end - start) * 1e-9
avg_time = total_time / 50
print(f"Fatto. Tempo totale: {total_time:.2f} secondi")
print(f"Latenza media per inferenza: {avg_time:.6f} s")
