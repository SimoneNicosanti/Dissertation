import copy
import itertools
import json
import multiprocessing
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import onnx
import onnxruntime
import pandas as pd
import tqdm
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.calibrate import create_calibrator
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.quant_utils import load_model_with_shape_infer
from onnxruntime.quantization.registry import QDQRegistry, QLinearOpsRegistry
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

MODEL_NAME = "yolo11n-seg"


COCO_FILE_PATH = "../../../coco128/preprocessed"

MAX_QUANTIZABLE_LAYERS = 10

TRAIN_SIZE = 750
TEST_SIZE = 50

CALIBRATION_DATA_SIZE = 100
CALIBRATION_TEST_SIZE = 20

PROCESSES_NUM = 1


class MyDataReader(CalibrationDataReader):

    def __init__(self, calibration_data: list[np.ndarray]):
        self.idx = 0
        self.tot_data = len(calibration_data)
        self.calibration_data = calibration_data
        self.generator = np.random.default_rng(seed=1)

    def get_next(self):
        if self.idx == self.tot_data:
            return None
        input = self.calibration_data[self.idx]
        self.idx += 1
        return {"images": input}


def quantize_model(
    subModelName,
    nodesToQuantize: list[str],
    calibration_data: list[np.ndarray],
    proc_idx: int,
):

    quantize_static(
        model_input=subModelName + f"_pre_quant_{proc_idx}.onnx",
        model_output=subModelName + f"_quant_{proc_idx}.onnx",
        quant_format=QuantFormat.QDQ,
        calibration_data_reader=MyDataReader(calibration_data),
        nodes_to_quantize=nodesToQuantize,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
    )


def quantize_model_1(
    model: onnx.ModelProto, tensors_range, nodes_to_quantize, op_types_to_quantize
):

    model_copy = copy.deepcopy(model)

    quantized_model = QDQQuantizer(
        model=model_copy,
        per_channel=False,
        reduce_range=False,
        weight_qType=QuantType.QInt8,
        activation_qType=QuantType.QInt8,
        tensors_range=tensors_range,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=[],
        op_types_to_quantize=op_types_to_quantize,
        extra_options={},
    ).quantize_model()

    return quantized_model
    pass


def test_model(
    test_dataset: np.ndarray, model_path: str, onnx_model: onnx.ModelProto = None
):
    options = onnxruntime.SessionOptions()
    # Imposta un solo thread intra-op (esecuzione interna a un'op)
    # options.intra_op_num_threads = 1
    # Imposta un solo thread inter-op (tra op diverse)
    # options.inter_op_num_threads = 1

    if onnx_model is None:
        session = onnxruntime.InferenceSession(model_path, sess_options=options)
    else:
        model_bytes = onnx_model.SerializeToString()
        session = onnxruntime.InferenceSession(model_bytes, sess_options=options)

    values = []
    for i in range(len(test_dataset)):
        input = test_dataset[i]
        value = session.run(None, {"images": input})
        values.append(value)

    del session

    return values


def build_model_graph():
    with open(f"{MODEL_NAME}.json") as f:
        data = json.load(f)

    # Converte il JSON in grafo NetworkX
    model_graph = nx.readwrite.node_link_graph(data)
    return model_graph


def read_all_images():

    files = os.listdir(COCO_FILE_PATH)

    # Elenca solo i file (non le directory)
    files = [f for f in files if os.path.isfile(os.path.join(COCO_FILE_PATH, f))]
    images = []
    for file_name in files:
        file_path = os.path.join(COCO_FILE_PATH, file_name)
        image = np.load(file_path)["arr_0"]

        images.append(image)

    return images


def process_function(
    proc_idx: int,
    cases: list[np.ndarray],
    quantizable_layers: list[str],
    calibration_dataset: list[np.ndarray],
    test_dataset: list[np.ndarray],
    not_quant_values: list[np.ndarray],
    queue: multiprocessing.Queue,
):
    op_types_to_quantize = list(
        set(list(QLinearOpsRegistry.keys()) + list(QDQRegistry.keys()))
    )

    quant_pre_process(
        MODEL_NAME + ".onnx",
        output_model_path=MODEL_NAME + f"_pre_quant_{proc_idx}.onnx",
    )

    calibrator = create_calibrator(
        MODEL_NAME + f"_pre_quant_{proc_idx}.onnx",
        augmented_model_path=MODEL_NAME + f"_augmented_{proc_idx}.onnx",
        use_external_data_format=False,
        extra_options={},
    )
    calibrator.collect_data(MyDataReader(calibration_dataset))
    tensors_range = calibrator.compute_data()

    model = load_model_with_shape_infer(
        Path(MODEL_NAME + f"_pre_quant_{proc_idx}.onnx")
    )

    Y_data = []

    for i in tqdm.tqdm(range(len(cases))):

        # print(f"ProcIdx {proc_idx} >> Case {i}")
        curr_case = cases[i]
        curr_quant_layers = []
        for j in range(MAX_QUANTIZABLE_LAYERS):
            if curr_case[j] == 1:
                curr_quant_layers.append(quantizable_layers[j])

        start = time.perf_counter_ns()
        quantized_model = quantize_model_1(
            model, tensors_range, curr_quant_layers, op_types_to_quantize
        )
        end = time.perf_counter_ns()
        # print("Quantization Time >> ", (end - start) / 1e9)

        start = time.perf_counter_ns()
        quant_values = test_model(
            test_dataset, f"{MODEL_NAME}_quant_{proc_idx}.onnx", quantized_model
        )
        end = time.perf_counter_ns()
        # print("Inference Time >> ", (end - start) / 1e9)

        noise = 0
        for i in range(len(not_quant_values)):

            quant_val = quant_values[i]
            not_quant_val = not_quant_values[i]

            curr_noise = 0
            for j in range(len(quant_val)):
                curr_noise += np.mean(
                    np.abs(quant_val[j] - not_quant_val[j]),
                    axis=None,
                )
            noise += curr_noise / len(quant_val)

        noise = noise / len(not_quant_values)
        Y_data.append(noise)

    os.remove(MODEL_NAME + f"_pre_quant_{proc_idx}.onnx")
    os.remove(MODEL_NAME + f"_augmented_{proc_idx}.onnx")

    queue.put(Y_data)


def build_data():

    all_images = read_all_images()
    calibration_dataset = all_images[:CALIBRATION_DATA_SIZE]
    test_dataset = all_images[
        CALIBRATION_DATA_SIZE : CALIBRATION_DATA_SIZE + CALIBRATION_TEST_SIZE
    ]

    not_quant_values = test_model(test_dataset, f"{MODEL_NAME}.onnx")

    model_graph = build_model_graph()

    sorted_by_flops = sorted(
        model_graph.nodes,
        key=lambda x: model_graph.nodes[x]["flops"],
        reverse=True,
    )
    quantizable_layers = sorted_by_flops[:MAX_QUANTIZABLE_LAYERS]

    for elem in quantizable_layers:
        print(elem, model_graph.nodes[elem]["flops"])

    quantizable_layers.sort()  ## Sorted by name

    X_data_set = set()
    X_data_set.add(tuple([1] * len(quantizable_layers)))

    X_data = []
    X_data.append([1] * len(quantizable_layers))

    random_generator = np.random.default_rng(seed=1)

    if TRAIN_SIZE + TEST_SIZE >= 2**MAX_QUANTIZABLE_LAYERS:
        for elem in itertools.product([0, 1], repeat=len(quantizable_layers)):
            X_data.append(list(elem))
    else:
        for _ in range(TRAIN_SIZE + TEST_SIZE - 1):
            random_quant = random_generator.integers(0, 2, len(quantizable_layers))

            while tuple(random_quant) in X_data_set or np.sum(random_quant) == 0:
                random_quant = random_generator.integers(0, 2, len(quantizable_layers))

            X_data_set.add(tuple(random_quant))
            X_data.append(random_quant)

    Y_data = []

    processes = []
    queues = []
    work_size = int((TRAIN_SIZE + TEST_SIZE) / PROCESSES_NUM)
    for proc_idx in range(PROCESSES_NUM):
        queue = multiprocessing.Queue()
        queues.append(queue)

        proc = multiprocessing.Process(
            target=process_function,
            args=(
                proc_idx,
                X_data[work_size * proc_idx : work_size * (proc_idx + 1)],
                quantizable_layers,
                calibration_dataset,
                test_dataset,
                not_quant_values,
                queue,
            ),
        )

        processes.append(proc)

    for proc in processes:
        proc.start()

    for idx, proc in enumerate(processes):
        proc.join()

        queue = queues[idx]
        while not queue.empty():
            Y_data.extend(queue.get())

    total_data = []
    for i in range(work_size * PROCESSES_NUM):
        total_data.append(np.concatenate((X_data[i], [Y_data[i]])))

    final_dataframe = pd.DataFrame(total_data, columns=quantizable_layers + ["Error"])
    final_dataframe.to_csv(f"data_{MODEL_NAME}_{MAX_QUANTIZABLE_LAYERS}.csv")


def build_predictor():

    dataframe = pd.read_csv(f"data_{MODEL_NAME}_{MAX_QUANTIZABLE_LAYERS}.csv")
    dataframe.drop("Unnamed: 0", axis=1, inplace=True)

    train_data = dataframe[:TRAIN_SIZE]
    test_data = dataframe[TRAIN_SIZE:]

    X_train, Y_train = train_data.drop("Error", axis=1), train_data["Error"]
    X_test, Y_test = test_data.drop("Error", axis=1), test_data["Error"]

    max_degree = 3

    # Crea figure con 2 subplot orizzontali
    fig, axes = plt.subplots(max_degree, 2, figsize=(12, 16))  # 1 riga, 2 colonne

    max_true_value = Y_train[0]

    test_scores = []
    models = []
    for degree in range(1, max_degree + 1):
        print("Degree >> ", degree)
        curr_axes = axes[degree - 1]

        poly_degree = degree
        model = Pipeline(
            [
                (
                    "poly_features",
                    PolynomialFeatures(
                        degree=poly_degree,
                        include_bias=False,
                        interaction_only=True,
                    ),
                ),
                ("lin_reg", LinearRegression()),
            ]
        )

        # Addestramento
        model.fit(X_train, Y_train)
        models.append(model)

        train_score = model.score(X_train, Y_train)
        test_score = model.score(X_test, Y_test)

        test_scores.append(test_score)

        # Previsioni
        predictions_test = model.predict(X_test)
        predictions_train = model.predict(X_train)

        # Limite per la diagonale
        line_limit = (
            max(
                Y_test.max(),
                Y_train.max(),
                predictions_test.max(),
                predictions_train.max(),
            )
            + 0.05
        )

        # Train plot
        curr_axes[0].scatter(Y_train, predictions_train, s=5, color="blue")
        curr_axes[0].plot(
            np.arange(0, line_limit, 0.05),
            np.arange(0, line_limit, 0.05),
            color="orange",
        )
        curr_axes[0].set_title(f"Train - Degree {degree} - Score {train_score:.4f}")
        curr_axes[0].set_xlabel("True Values")
        curr_axes[0].set_ylabel("Predictions")

        ## Test Plot
        curr_axes[1].scatter(Y_test, predictions_test, s=5, color="green")
        curr_axes[1].plot(
            np.arange(0, line_limit, 0.05),
            np.arange(0, line_limit, 0.05),
            color="orange",
        )
        curr_axes[1].set_title(f"Test - Degree {degree} - Score {test_score:.4f} ")
        curr_axes[1].set_xlabel("True Values")
        curr_axes[1].set_ylabel("Predictions")

        curr_axes[0].axvline(
            x=max_true_value, color="red", linestyle="--", linewidth=2
        )  # Linea verticale a x=2

    plt.tight_layout()
    plt.savefig(f"regression_plot_{MODEL_NAME}.png")
    plt.clf()

    best_model_idx = int(np.argmax(test_scores))
    best_model = models[best_model_idx]

    print("Best Model Degree >> ", best_model_idx + 1)

    # Estrai lo step PolynomialFeatures dalla pipeline
    poly = best_model.named_steps["poly_features"]

    # Ottieni i nomi delle feature polinomiali
    feature_names = poly.get_feature_names_out(input_features=X_train.columns)
    feature_tuples = [tuple(name.split(" ")) for name in feature_names]

    # Ora puoi stampare i coefficienti con i nomi
    lin_reg = best_model.named_steps["lin_reg"]
    coeffs = lin_reg.coef_

    model_repr = {}
    model_repr["coef"] = {}
    model_repr["intercept"] = {}

    # Stampa ordinata
    for name, coef in zip(feature_tuples, coeffs):
        model_repr["coef"][str(name)] = coef
    model_repr["intercept"] = lin_reg.intercept_

    with open(f"error_model_{MODEL_NAME}.json", "w") as f:
        json.dump(model_repr, f, sort_keys=True)


def modify_model():
    model_graph = build_model_graph()

    sorted_by_flops = sorted(
        model_graph.nodes,
        key=lambda x: model_graph.nodes[x]["flops"],
        reverse=True,
    )
    quantizable_layers = []
    while len(quantizable_layers) < MAX_QUANTIZABLE_LAYERS:
        quantizable_layers.append(sorted_by_flops.pop(0))

    for layer in quantizable_layers:
        model_graph.nodes[layer]["quantizable"] = True

    data = nx.readwrite.json_graph.node_link_data(model_graph)

    # Esportare il grafo come JSON su un file
    with open("graph.json", "w") as f:
        json.dump(
            data,
            f,
        )


def post_process(output):
    predictions = np.squeeze(output, axis=0)
    predictions = predictions.T

    predictions = predictions[:, :84]
    boxes = predictions[:, :4]
    classes = predictions[:, 4:]

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    boxes[:, 0] = x - w / 2  # x1
    boxes[:, 1] = y - h / 2  # y1
    boxes[:, 2] = x + w / 2  # x2
    boxes[:, 3] = y + h / 2  # y2

    scores = np.max(classes, axis=1)
    class_ids = np.argmax(classes, axis=1)

    final = np.concatenate(
        [
            boxes,
            class_ids.reshape(-1, 1),
            scores.reshape(-1, 1),
        ],
        axis=1,
    )

    return final


if __name__ == "__main__":
    build_data()

    build_predictor()

    # modify_model()
