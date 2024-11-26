import keras
import tensorflow as tf
from tensorflow.core.profiler.tfprof_output_pb2 import GraphNodeProto
from tensorflow.python.profiler import model_analyzer, option_builder

model: keras.Model = keras.applications.MobileNetV3Large()

for layer in model.layers:
    # print(layer.input, layer.output)

    if isinstance(layer, keras.layers.InputLayer):
        continue

    subModel = keras.Model(inputs=layer.input, outputs=layer.output)
# for layer in model.layers:
#     if isinstance(layer, keras.layers.InputLayer):
#         continue

#     subModel = keras.Model(inputs=layer.input, outputs=layer.output)

#     input_signature = [
#         tf.TensorSpec(shape=(1, 224, 224, 3), dtype=params.dtype, name=params.name)
#         for params in subModel.inputs
#     ]

#     forward_graph = tf.function(subModel, input_signature).get_concrete_function().graph
#     options = option_builder.ProfileOptionBuilder.float_operation()
#     graph_info = model_analyzer.profile(forward_graph, options=options)
#     flops = graph_info.total_float_ops

#     print(f"{layer.name} FLOPS >>> {flops}\n")


input_signature = [
    tf.TensorSpec(shape=(1, 32, 32, 3), dtype=params.dtype, name=params.name)
    for params in model.inputs
]

forward_graph = tf.function(model, input_signature).get_concrete_function().graph
options = option_builder.ProfileOptionBuilder.float_operation()

graph_info: GraphNodeProto = model_analyzer.profile(forward_graph, options=options)
flops = graph_info.total_float_ops

opsDict = {}
for child in graph_info.children:
    childName: str = child.name
    childNameParts: list = childName.split("/")  ## First Part is model name
    levelName: str = childNameParts[1]

    if levelName not in opsDict:
        opsDict[levelName] = 0
    opsDict[levelName] += child.total_float_ops


total = 0
for key in opsDict:
    total += opsDict[key]
print(f"TOTAL MODEL FLOPS >>> {flops} {total}")  ## 448572872
