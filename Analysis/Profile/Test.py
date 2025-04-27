import csv
import json
import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnx_tool
import onnxruntime as ort
import pandas
from onnx import TensorProto, helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm


def create_conv_onnx(kernel_size, input_channels, output_channels, input_size):
    """Crea un modello ONNX con un singolo strato convoluzionale."""
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, input_channels, input_size, input_size]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, output_channels, -1, -1]
    )

    weight_shape = [output_channels, input_channels, kernel_size, kernel_size]
    weight = np.random.randn(*weight_shape).astype(np.float32)
    weight_initializer = helper.make_tensor(
        "conv_weight", TensorProto.FLOAT, weight_shape, weight.flatten()
    )

    conv_node = helper.make_node(
        "Conv",
        ["input", "conv_weight"],
        ["output"],
        kernel_shape=[kernel_size, kernel_size],
        name="Conv",
    )

    graph = helper.make_graph(
        [conv_node], "ConvModel", [input_tensor], [output_tensor], [weight_initializer]
    )
    model = helper.make_model(
        graph,
        producer_name="onnx-example",
        opset_imports=[helper.make_operatorsetid("", 21)],
    )

    onnx.save(model, "conv.onnx")
    return model


def measure_execution_time(model_path, input_shape, runs):
    """Misura il tempo medio di esecuzione del modello con onnxruntime."""
    session_options = ort.SessionOptions()
    session_options.enable_profiling = True

    session = ort.InferenceSession(
        model_path, providers=["CPUExecutionProvider"], sess_options=session_options
    )
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    times = []
    for _ in range(runs):
        time.perf_counter_ns()
        session.run(None, {"input": dummy_input})
        # times.append((time.perf_counter_ns() - start) / 1e9)

    json_file_name = session.end_profiling()

    with open(json_file_name, "r") as json_file:
        data_list = json.load(json_file)
        elem: str
        for elem in data_list:
            if (
                elem["name"].find("kernel_time") != -1
                and elem["args"]["op_name"].find("Conv") != -1
            ):
                times.append(elem["dur"] / 1e6)

    os.remove(json_file_name)
    print(np.mean(times))
    return np.mean(times)


def measure_flops(model_path, input_shape):
    model = onnx.load_model(model_path)
    tool_model = onnx_tool.Model(m=model)

    input = np.random.random(size=input_shape)
    tool_model.graph.shape_infer({"input": input})
    tool_model.graph.profile()

    tempfile_name: str = tempfile.mktemp() + ".csv"
    tool_model.graph.print_node_map(tempfile_name, metric="FLOPs")
    with open(tempfile_name, "r") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0 or row[0] == "Total":
                continue
            return float(row[2])


def measure_quantized_execution_time(model_path, input_shape, runs):
    class DummyCalibrationDataReader(CalibrationDataReader):
        def __init__(self, data):
            self.data = data
            self.index = 0

        def get_next(self):
            if self.index < len(self.data):
                batch = self.data[self.index]
                self.index += 1
                return {"input": batch}
            return None

        def reset(self):
            self.index = 0

    # Esempio di dati di calibrazione (usa i tuoi dati reali)
    calibration_data = [
        np.random.rand(*input_shape).astype(np.float32) for _ in range(1)
    ]  # Dummy data

    # Crea un oggetto di calibrazione personalizzato
    calibration_reader = DummyCalibrationDataReader(calibration_data)

    quant_pre_process(model_path, output_model_path="conv_pre_quantized.onnx")

    # Quantizzazione statica del modello (INT8)
    quantize_static(
        "conv_pre_quantized.onnx",  # il modello pre-elaborato
        model_output="conv_quantized.onnx",  # il modello quantizzato
        calibration_data_reader=calibration_reader,  # il lettore dei dati di calibrazione
        weight_type=QuantType.QUInt8,  # Tipo di quantizzazione per i pesi
        activation_type=QuantType.QUInt8,  # Tipo di quantizzazione per le attivazioni
    )

    return measure_execution_time("conv_quantized.onnx", input_shape, runs)


# Parametri della rete

generator = np.random.default_rng(0)

done_tests = set()

num_models = 200
runs_per_model = 3

flops_list = []
times_list = []
quan_times_list = []

for _ in tqdm(range(num_models // 3)):
    kernel = generator.integers(2, 9)
    in_ch = generator.integers(3, 17)
    out_ch = generator.integers(3, 17)
    input_size = generator.integers(32, 257)

    if (input_size, kernel, in_ch, out_ch) in done_tests:
        continue
    done_tests.add((input_size, kernel, in_ch, out_ch))

    model = create_conv_onnx(int(kernel), int(in_ch), int(out_ch), int(input_size))

    flops = measure_flops("conv.onnx", [1, in_ch, input_size, input_size])

    time_exec = measure_execution_time(
        "conv.onnx", [1, in_ch, input_size, input_size], runs_per_model
    )
    quantized_time_exec = measure_quantized_execution_time(
        "conv.onnx", [1, in_ch, input_size, input_size], runs_per_model
    )

    flops_list.append(flops)
    times_list.append(time_exec)
    quan_times_list.append(quantized_time_exec)

for _ in tqdm(range(num_models // 3)):
    kernel = generator.integers(9, 17)
    in_ch = generator.integers(17, 33)
    out_ch = generator.integers(17, 33)
    input_size = generator.integers(257, 513)

    if (input_size, kernel, in_ch, out_ch) in done_tests:
        continue
    done_tests.add((input_size, kernel, in_ch, out_ch))

    model = create_conv_onnx(int(kernel), int(in_ch), int(out_ch), int(input_size))

    flops = measure_flops("conv.onnx", [1, in_ch, input_size, input_size])

    time_exec = measure_execution_time(
        "conv.onnx", [1, in_ch, input_size, input_size], runs_per_model
    )
    quantized_time_exec = measure_quantized_execution_time(
        "conv.onnx", [1, in_ch, input_size, input_size], runs_per_model
    )

    flops_list.append(flops)
    times_list.append(time_exec)
    quan_times_list.append(quantized_time_exec)

for _ in tqdm(range(num_models // 3)):
    kernel = generator.integers(9, 18)
    in_ch = generator.integers(17, 33)
    out_ch = generator.integers(17, 33)
    input_size = generator.integers(513, 751)

    if (input_size, kernel, in_ch, out_ch) in done_tests:
        continue
    done_tests.add((input_size, kernel, in_ch, out_ch))

    model = create_conv_onnx(int(kernel), int(in_ch), int(out_ch), int(input_size))

    flops = measure_flops("conv.onnx", [1, in_ch, input_size, input_size])

    time_exec = measure_execution_time(
        "conv.onnx", [1, in_ch, input_size, input_size], runs_per_model
    )
    quantized_time_exec = measure_quantized_execution_time(
        "conv.onnx", [1, in_ch, input_size, input_size], runs_per_model
    )

    flops_list.append(flops)
    times_list.append(time_exec)
    quan_times_list.append(quantized_time_exec)


data = {
    "flops": flops_list + flops_list,  # Lista di flops
    "is_quantized": (
        [0] * len(flops_list)  # 1 per modelli quantizzati
        + [1] * len(flops_list)  # 0 per modelli non quantizzati
    ),
    "execution_time": times_list
    + quan_times_list,  # Tempi di esecuzione (non quantizzato + quantizzato)
}

# Crea un DataFrame con i dati
df = pandas.DataFrame(data)

# Separiamo i dati in due sottoinsiemi
df_quantized = df[df["is_quantized"] == 1]
df_non_quantized = df[df["is_quantized"] == 0]


# Funzione per eseguire la regressione polinomiale
def polynomial_regression(X, y, degree=2):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    regressor = LinearRegression()
    regressor.fit(X_poly, y)
    return poly, regressor


# Eseguiamo la regressione separata per modelli quantizzati e non quantizzati
X_quantized = df_quantized[["flops", "is_quantized"]]
y_quantized = df_quantized["execution_time"]
poly_quantized, regressor_quantized = polynomial_regression(X_quantized, y_quantized)

X_non_quantized = df_non_quantized[["flops", "is_quantized"]]
y_non_quantized = df_non_quantized["execution_time"]
poly_non_quantized, regressor_non_quantized = polynomial_regression(
    X_non_quantized, y_non_quantized
)

# Generiamo i punti per la visualizzazione della regressione
flops_range = np.linspace(min(df["flops"]), max(df["flops"]), 100)

# Creiamo i punti di input per la previsione
X_vis_quantized = np.array([[flop, 1] for flop in flops_range])  # is_quantized = 1
X_vis_non_quantized = np.array([[flop, 0] for flop in flops_range])  # is_quantized = 0

# Trasformiamo i punti in termini polinomiali
X_vis_poly_quantized = poly_quantized.transform(X_vis_quantized)
X_vis_poly_non_quantized = poly_non_quantized.transform(X_vis_non_quantized)

# Previsioni della regressione
y_vis_quantized = regressor_quantized.predict(X_vis_poly_quantized)
y_vis_non_quantized = regressor_non_quantized.predict(X_vis_poly_non_quantized)

# ---- Calcolo delle differenze tra quantizzati e non quantizzati ----
df_diff = df_quantized.copy()
df_diff["execution_time_diff"] = (
    df_quantized["execution_time"].values - df_non_quantized["execution_time"].values
)

y_vis_diff = y_vis_quantized - y_vis_non_quantized

# ---- Creazione del grafico ----
fig, axes = plt.subplots(2, 1, figsize=(10, 18))

# ---- Grafico della regressione ----
axes[0].plot(
    flops_range,
    y_vis_quantized,
    label="Regressione Quantizzata",
    color="red",
    linewidth=2,
)
axes[0].plot(
    flops_range,
    y_vis_non_quantized,
    label="Regressione Non Quantizzata",
    color="blue",
    linewidth=2,
)
axes[0].scatter(
    df["flops"],
    df["execution_time"],
    c=df["is_quantized"],
    cmap="viridis",
    label="Dati Originali",
    alpha=0.6,
)
axes[0].set_xlabel("Flops")
axes[0].set_ylabel("Tempo di esecuzione")
axes[0].set_title("Regressione Polinomiale: Flops vs Tempo di Esecuzione")
axes[0].legend()

# ---- Grafico delle differenze tra quantizzati e non quantizzati ----
axes[1].scatter(
    df_quantized["flops"],
    df_diff["execution_time_diff"],
    color="purple",
    label="Differenza Quantizzati - Non Quantizzati",
)
axes[1].set_xlabel("Flops")
axes[1].set_ylabel("Differenza Tempo di Esecuzione")
axes[1].set_title("Differenza tra Tempi di Esecuzione (Quantizzati - Non Quantizzati)")
axes[1].legend()

# ---- Grafico della differenza tra le funzioni di regressione ----
axes[1].plot(
    flops_range,
    y_vis_diff,
    label="Differenza tra le regressioni",
    color="green",
    linewidth=2,
)
axes[1].fill_between(
    flops_range,
    y_vis_diff - 0.05,
    y_vis_diff + 0.05,
    color="green",
    alpha=0.2,
)
axes[1].set_xlabel("Flops")
axes[1].set_ylabel("Differenza Tempo di Esecuzione Previsto")
axes[1].set_title("Differenza tra le Funzioni di Regressione")
axes[1].legend()

plt.tight_layout()
plt.savefig("regression_results.png")


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Separiamo i dati in training set e test set
X_train_quantized, X_test_quantized, y_train_quantized, y_test_quantized = (
    train_test_split(X_quantized, y_quantized, test_size=0.2, random_state=42)
)
(
    X_train_non_quantized,
    X_test_non_quantized,
    y_train_non_quantized,
    y_test_non_quantized,
) = train_test_split(X_non_quantized, y_non_quantized, test_size=0.2, random_state=42)

# Regressione polinomiale sui dati di addestramento
poly_quantized, regressor_quantized = polynomial_regression(
    X_train_quantized, y_train_quantized
)
poly_non_quantized, regressor_non_quantized = polynomial_regression(
    X_train_non_quantized, y_train_non_quantized
)

# Eseguiamo le previsioni sui dati di test
y_pred_quantized = regressor_quantized.predict(
    poly_quantized.transform(X_test_quantized)
)
y_pred_non_quantized = regressor_non_quantized.predict(
    poly_non_quantized.transform(X_test_non_quantized)
)

# Calcoliamo l'errore medio quadratico (RMSE) per entrambi i modelli
rmse_quantized = calculate_rmse(y_test_quantized, y_pred_quantized)
rmse_non_quantized = calculate_rmse(y_test_non_quantized, y_pred_non_quantized)

print(f"RMSE per il modello quantizzato: {rmse_quantized}")
print(f"RMSE per il modello non quantizzato: {rmse_non_quantized}")

# Aggiungi i test su entrambe le regressioni nel grafico
# Ora possiamo generare anche il grafico delle previsioni di test

# ---- Grafico delle previsioni di test ----
fig, axes = plt.subplots(2, 1, figsize=(10, 18))

# ---- Grafico delle previsioni di test ----
axes[0].scatter(
    X_test_quantized.iloc[:, 0],
    y_test_quantized,
    color="red",
    label="Test Quantizzato (Reale)",
)
axes[0].scatter(
    X_test_quantized.iloc[:, 0],
    y_pred_quantized,
    color="orange",
    label="Test Quantizzato (Predetto)",
)
axes[0].scatter(
    X_test_non_quantized.iloc[:, 0],
    y_test_non_quantized,
    color="blue",
    label="Test Non Quantizzato (Reale)",
)
axes[0].scatter(
    X_test_non_quantized.iloc[:, 0],
    y_pred_non_quantized,
    color="cyan",
    label="Test Non Quantizzato (Predetto)",
)

axes[0].set_xlabel("Flops")
axes[0].set_ylabel("Tempo di esecuzione")
axes[0].set_title("Previsioni di Test: Quantizzato vs Non Quantizzato")
axes[0].legend()

plt.tight_layout()
plt.savefig("test_predictions.png")
