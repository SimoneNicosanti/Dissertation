import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto

# # Inputs
# input_tensor = helper.make_tensor_value_info(
#     "input", TensorProto.FLOAT, ["N", "H", "W", 3]
# )
# output_tensor = helper.make_tensor_value_info(
#     "output", TensorProto.FLOAT, ["N", 3, "out_H", "out_W"]
# )

# # Constants
# perm_tensor = numpy_helper.from_array(
#     np.array([2, 1, 0], dtype=np.int64), name="perm"
# )  # swap channels
# divisor_tensor = numpy_helper.from_array(
#     np.array(255.0, dtype=np.float32), name="divisor"
# )
# sizes_tensor = numpy_helper.from_array(
#     np.array([1, 640, 640, 3], dtype=np.int64), name="sizes"
# )

# # Nodes
# permute_rgb = helper.make_node(
#     "Gather", inputs=["input", "perm"], outputs=["rgb_image"], axis=3
# )

# resize = helper.make_node(
#     "Resize",
#     inputs=["rgb_image", "", "", "sizes"],  # empty inputs for 'roi' and 'scales'
#     outputs=["resized_image"],
#     mode="linear",
# )

# div = helper.make_node(
#     "Div", inputs=["resized_image", "divisor"], outputs=["normalized_image"]
# )

# transpose = helper.make_node(
#     "Transpose",
#     inputs=["normalized_image"],
#     outputs=["output"],  # Qui mettiamo l'output finale direttamente da Transpose
#     perm=[0, 3, 1, 2],  # Cambia da (N, H, W, C) a (N, C, H, W)
# )

# # Graph
# graph = helper.make_graph(
#     nodes=[permute_rgb, resize, div, transpose],
#     name="PreprocessingGraph",
#     inputs=[input_tensor],
#     outputs=[output_tensor],  # Output finale è quello di 'Transpose'
#     initializer=[perm_tensor, divisor_tensor, sizes_tensor],
# )

# # Model
# model = helper.make_model(graph, producer_name="custom-preprocessing")

# # Save the model
# onnx.save(model, "preprocessing_model.onnx")

input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 84, 8400])

squeeze_axes_zero = onnx.helper.make_tensor(
    "zero_squeeze_axes", TensorProto.INT64, dims=[1], vals=[0]
)

bbox_slice_starts = onnx.helper.make_tensor(
    "bbox_slice_starts", TensorProto.INT64, dims=[2], vals=[0, 0]
)
bbox_slice_ends = onnx.helper.make_tensor(
    "bbox_slice_ends", TensorProto.INT64, dims=[2], vals=[8400, 4]
)

class_score_slice_starts = onnx.helper.make_tensor(
    "class_score_slice_starts", TensorProto.INT64, dims=[2], vals=[0, 4]
)
class_score_slice_ends = onnx.helper.make_tensor(
    "class_score_slice_ends", TensorProto.INT64, dims=[2], vals=[8400, 84]
)

max_boxes = onnx.helper.make_tensor(
    "max_boxes", TensorProto.INT64, dims=[1], vals=[int(1e9)]
)
iou_threshold = onnx.helper.make_tensor(
    "iou_threshold", TensorProto.FLOAT, dims=[1], vals=[0.5]
)
score_threshold = onnx.helper.make_tensor(
    "score_threshold", TensorProto.FLOAT, dims=[1], vals=[0.8]
)

gather_node_idxs = onnx.helper.make_tensor(
    name="gather_node_idxs",
    data_type=TensorProto.INT64,
    dims=[1],
    vals=[2],
)

squeeze_axes_one = onnx.helper.make_tensor(
    name="squeeze_axes_one", data_type=TensorProto.INT64, dims=[1], vals=[1]
)


squeeze_node = onnx.helper.make_node(
    "Squeeze",
    inputs=[
        "input",
        squeeze_axes_zero.name,
    ],
    outputs=[
        "squeezed_tensor",
    ],
)

# Crea il nodo per trasporre il tensore (Transpose)
transpose_node = onnx.helper.make_node(
    "Transpose",
    inputs=["squeezed_tensor"],
    outputs=["transposed_tensor"],
    perm=[1, 0],  # Trasforma (84, 8400) in (8400, 84)
)

slice_node_4 = onnx.helper.make_node(
    "Slice",
    inputs=["transposed_tensor", bbox_slice_starts.name, bbox_slice_ends.name],
    outputs=["bbox_tensor"],
)

# Nodo per estrarre la parte 8400x80 (resto delle colonne)
slice_node_80 = onnx.helper.make_node(
    "Slice",
    inputs=[
        "transposed_tensor",
        class_score_slice_starts.name,
        class_score_slice_ends.name,
    ],
    outputs=["class_score_tensor"],
)

nms_node = onnx.helper.make_node(
    "NonMaxSuppression",  # Tipo di operatore
    inputs=[
        "bbox_tensor",
        "class_score_tensor",
        max_boxes.name,
        iou_threshold.name,
        score_threshold.name,
    ],  # Input: coordinate delle bounding box e punteggi
    outputs=[
        "selected_boxes",
    ],
)


gather_node = onnx.helper.make_node(
    "Gather",
    inputs=["selected_boxes", gather_node_idxs.name],
    outputs=["gathered_boxes"],
    axis=1,
)


squeeze_node_1 = onnx.helper.make_node(
    "Squeeze",
    inputs=[
        "gathered_boxes",
        squeeze_axes_one.name,
    ],
    outputs=["squeezed_boxes"],
)

gather_node_1 = onnx.helper.make_node(
    "Gather",
    inputs=["transposed_tensor", "squeezed_boxes"],
    outputs=["selected_data"],
    axis=0,  # Gathering sulle righe
)

slice_node_4_1 = onnx.helper.make_node(
    "Slice",
    inputs=["selected_data", bbox_slice_starts.name, bbox_slice_ends.name],
    outputs=["final_bbox_tensor"],
)

# Nodo per estrarre la parte 8400x80 (resto delle colonne)
slice_node_80_1 = onnx.helper.make_node(
    "Slice",
    inputs=[
        "selected_data",
        class_score_slice_starts.name,
        class_score_slice_ends.name,
    ],
    outputs=["final_classes_scores"],
)

max_score_axes = onnx.helper.make_tensor(
    name="max_score_axes", data_type=TensorProto.INT64, dims=[1], vals=[1]
)
max_score_node = onnx.helper.make_node(
    "ReduceMax",
    inputs=["final_classes_scores", max_score_axes.name],
    outputs=["max_score"],
    keepdims=1,
)

max_class_node = onnx.helper.make_node(
    "ArgMax",
    inputs=["final_classes_scores"],
    outputs=["max_class"],
    axis=1,
)

cast_node = onnx.helper.make_node(
    "Cast",
    inputs=["max_class"],
    outputs=["max_class_float"],
    to=TensorProto.FLOAT,  # Lo converte a float32
)

concat_node = onnx.helper.make_node(
    "Concat",
    inputs=["final_bbox_tensor", "max_score", "max_class_float"],
    outputs=["output"],
    axis=1,
)


nodes = [
    squeeze_node,
    transpose_node,
    slice_node_4,
    slice_node_80,
    nms_node,
    gather_node,
    squeeze_node_1,
    gather_node_1,
    slice_node_4_1,
    slice_node_80_1,
    max_score_node,
    max_class_node,
    cast_node,
    concat_node,
]


outputs = [
    onnx.helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [None, 6],
    ),  # (num_boxes_after_nms, 4)
]

graph = helper.make_graph(
    nodes,
    "PreprocessingGraph",
    inputs=[input_tensor],
    outputs=outputs,
    initializer=[
        squeeze_axes_zero,
        bbox_slice_starts,
        bbox_slice_ends,
        class_score_slice_starts,
        class_score_slice_ends,
        max_boxes,
        iou_threshold,
        score_threshold,
        gather_node_idxs,
        squeeze_axes_one,
        max_score_axes,
    ],
)

model = helper.make_model(graph, producer_name="custom-preprocessing")
onnx.checker.check_model(model, full_check=True)

onnx.save(model, "postprocessing_model.onnx")

model = onnx.load("postprocessing_model.onnx")

model = onnx.shape_inference.infer_shapes(model, data_prop=True)

onnx.save(model, "postprocessing_model.onnx")
