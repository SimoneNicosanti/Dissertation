import tempfile

import networkx as nx
import numpy as np
import onnx
import onnxruntime
import pandas as pd
import tqdm
from onnxruntime.quantization.calibrate import CalibrationDataReader

from CommonIds.NodeId import NodeId
from CommonProfile.ModelInfo import ModelNodeInfo
from CommonQuantization import SoftQuantization


class NoiseEvaluator:
    def __init__(self, model: onnx.ModelProto, noise_test_set: np.ndarray):

        self.noise_test_set = noise_test_set
        self.normal_results = self.compute_model_result(model)

        pass

    def compute_model_result(self, model: onnx.ModelProto) -> list[list[np.ndarray]]:
        so = onnxruntime.SessionOptions()
        so.log_severity_level = (
            3  # 0 = VERBOSE, 1 = INFO, 2 = WARNING, 3 = ERROR, 4 = FATAL
        )
        sess = onnxruntime.InferenceSession(
            model.SerializeToString(),
            # providers=[
            #     "CUDAExecutionProvider",
            #     # ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
            # ],
            sess_options=so,
        )

        results = []

        input_info = sess.get_inputs()[0]
        for i in range(len(self.noise_test_set)):
            elem = self.noise_test_set[i]
            elem_batch = np.expand_dims(elem, axis=0)
            input = {input_info.name: elem_batch}

            result = sess.run(None, input)
            results.append(result)

        return results

    def evaluate_noise(self, quantized_model: onnx.ModelProto) -> float:

        quantized_results = self.compute_model_result(quantized_model)

        total_noise = 0
        for res_list_idx in range(len(self.normal_results)):
            quant_res_list = quantized_results[res_list_idx]
            norm_res_list = self.normal_results[res_list_idx]

            curr_noise = 0
            for i in range(len(quant_res_list)):
                curr_noise += np.mean(
                    np.abs(quant_res_list[i] - norm_res_list[i]),
                    axis=None,
                )

            total_noise += curr_noise / len(quant_res_list)

        return total_noise / len(self.noise_test_set)


class DataReader(CalibrationDataReader):
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

        return None


class QuantizationProfile:
    def __init__(self):
        pass

    def profile_quantization(
        self,
        model: onnx.ModelProto,
        model_graph: nx.DiGraph,
        max_quantizable: int,
        calibration_dataset: np.ndarray,
        train_set_size: int,
        test_set_size: int,
        calibration_size: int,
        noise_test_size: int,
    ) -> pd.DataFrame:

        quantizable_layers: list[NodeId] = self.find_quantizable_layers(
            model_graph, max_quantizable
        )
        self.mark_layers(model_graph, quantizable_layers)

        _, temp_file = tempfile.mkstemp(suffix=".onnx")
        onnx.save_model(model, temp_file)

        points = self.build_points(quantizable_layers, train_set_size, test_set_size)

        noises = self.compute_quantization_noises(
            temp_file,
            points,
            quantizable_layers,
            calibration_dataset,
            calibration_size,
            noise_test_size,
        )

        combined = [point + [noise] for point, noise in zip(points, noises)]
        cols_names = [node_id.node_name for node_id in quantizable_layers] + ["noise"]
        dataframe = pd.DataFrame(data=combined, columns=cols_names)
        print("Dataframe Built")

        return dataframe

        pass

    def find_quantizable_layers(
        self, model_graph: nx.DiGraph, max_quantizable: int
    ) -> list[NodeId]:

        layer_info_list = []
        for node_id in model_graph.nodes:
            layer_info_list.append((node_id, model_graph.nodes[node_id]["flops"]))

        layer_info_list.sort(key=lambda x: x[1], reverse=True)

        layer_info_list = layer_info_list[:max_quantizable]

        return [layer_info[0] for layer_info in layer_info_list]

    def mark_layers(self, model_graph: nx.DiGraph, quantizable_layers: list[str]):
        for layer in quantizable_layers:
            model_graph.nodes[layer][ModelNodeInfo.QUANTIZABLE] = True

    def build_points(self, quantizable_layers, train_set_size, test_set_size):
        total_size = train_set_size + test_set_size

        if total_size > 2 ** len(quantizable_layers) - 1:
            raise Exception("Too Many Data Points")

        random_generator = np.random.default_rng(seed=2)

        points = []
        while len(points) < total_size:
            if len(points) == 0:
                point = tuple([1] * len(quantizable_layers))
            else:
                point = tuple(
                    random_generator.integers(0, 2, size=len(quantizable_layers))
                )

            if point not in points and sum(point) > 0:
                points.append(tuple(point))

        for i in range(len(points)):
            points[i] = list(points[i])
        return points

    def compute_quantization_noises(
        self,
        model_path: str,
        points: list[tuple],
        quantizable_layers: list[NodeId],
        calibration_dataset: np.ndarray,
        calibration_set_size: int,
        noise_test_size: int,
    ) -> list[float]:

        noise_test_set = calibration_dataset[
            calibration_set_size : calibration_set_size + noise_test_size
        ]
        noise_evaluator = NoiseEvaluator(onnx.load_model(model_path), noise_test_set)

        calibration_set = calibration_dataset[:calibration_set_size]
        calibration_data_reader = DataReader(model_path, calibration_set)

        model, tensors_range = SoftQuantization.prepare_quantization(
            model_path, calibration_data_reader=calibration_data_reader
        )

        noises = []
        for idx, point in tqdm.tqdm(enumerate(points)):
            # print("Progress >> ", idx, "/", len(points))
            noise = self.compute_quantization_noise(
                model, tensors_range, point, quantizable_layers, noise_evaluator
            )
            noises.append(noise)

        return noises

    def compute_quantization_noise(
        self,
        model: onnx.ModelProto,
        tensors_range,
        point: tuple,
        quantizable_layers: list[NodeId],
        noise_evaluator: NoiseEvaluator,
    ) -> None:
        selected_layers: list[NodeId] = np.extract(
            condition=point, arr=quantizable_layers
        )
        selected_layers = [node_id.node_name for node_id in selected_layers]
        quantized_model = SoftQuantization.soft_quantization(
            model, tensors_range, nodes_to_quantize=selected_layers
        )

        return noise_evaluator.evaluate_noise(quantized_model)
        pass
