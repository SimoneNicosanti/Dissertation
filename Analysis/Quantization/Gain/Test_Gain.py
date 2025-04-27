import json
import time

import networkx as nx
import numpy as np
import onnxruntime
import tqdm
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

LAYER_LIST = [
    "/model.23/proto/cv2/conv/Conv",
    "/model.23/proto/upsample/ConvTranspose",
    "/model.3/conv/Conv",
    "/model.23/proto/cv1/conv/Conv",
    "/model.5/conv/Conv",
    "/model.1/conv/Conv",
    "/model.2/cv2/conv/Conv",
    "/model.4/cv2/conv/Conv",
    "/model.16/cv1/conv/Conv",
    "/model.17/conv/Conv",
    "/model.23/cv4.0/cv4.0.0/conv/Conv",
    "/model.23/cv2.0/cv2.0.0/conv/Conv",
    "/model.20/conv/Conv",
    "/model.7/conv/Conv",
    "/model.16/cv2/conv/Conv",
    "/model.19/cv2/conv/Conv",
    "/model.6/cv2/conv/Conv",
    "/model.13/cv1/conv/Conv",
    "/model.13/cv2/conv/Conv",
    "/model.19/cv1/conv/Conv",
]


def build_model_graph():
    with open("yolo11x-seg.json") as f:
        data = json.load(f)

    # Converte il JSON in grafo NetworkX
    model_graph = nx.readwrite.node_link_graph(data)
    return model_graph


class MyDataReader:

    def __init__(self):
        self.idx = 0
        self.tot_data = 1

    def get_next(self):
        if self.idx == self.tot_data:
            return None
        input = np.ones((1, 3, 640, 640), dtype=np.float32)
        self.idx += 1
        return {"images": input}


def quantize_model(subModelName, nodesToQuantize: list[str]):

    quantize_static(
        model_input=subModelName + "_pre_quant.onnx",
        model_output=subModelName + "_quant.onnx",
        quant_format=QuantFormat.QDQ,
        calibration_data_reader=MyDataReader(),
        nodes_to_quantize=nodesToQuantize,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
    )


def test_model(model_path: str, test_num: int):
    session = onnxruntime.InferenceSession(model_path)

    input = np.ones((1, 3, 640, 640), dtype=np.float32)
    session.run(None, {"images": input})

    start = time.perf_counter_ns()
    for _ in tqdm.tqdm(range(0, test_num)):
        session.run(None, {"images": input})
    end = time.perf_counter_ns()

    return (end - start) * 1e-9 / test_num


def main_1():
    model_graph = build_model_graph()
    quantizable_layers = []
    flops_list = []
    for layer in model_graph.nodes:
        flops_list.append((layer, model_graph.nodes[layer]["flops"]))
    flops_list.sort(key=lambda x: x[1], reverse=True)

    top_flops_layers = flops_list[:20]
    quantizable_layers = [flops[0] for flops in top_flops_layers]
    print(quantizable_layers)
    pass


def main_2():

    main_model_path = "yolo11x-seg.onnx"

    # model_time = test_model(
    #     model_path=main_model_path,
    #     test_num=50,
    # )
    # print("Not Quant Time >> ", model_time)

    quant_pre_process(
        main_model_path,
        output_model_path="yolo11x-seg_pre_quant.onnx",
    )

    quantize_model(subModelName="yolo11x-seg", nodesToQuantize=LAYER_LIST)

    quant_model_time = test_model(
        model_path="yolo11x-seg_quant.onnx",
        test_num=50,
    )
    print("Quant Time >> ", quant_model_time)


if __name__ == "__main__":
    # main_1()

    main_2()
