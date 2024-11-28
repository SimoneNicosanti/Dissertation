import keras
import tensorflow as tf
from tensorflow.python.profiler import model_analyzer, option_builder

model: keras.Model = keras.applications.MobileNetV3Large()

input_signature = [
    tf.TensorSpec(shape=(1, 32, 32, 3), dtype=params.dtype, name=params.name)
    for params in model.inputs
]

forward_graph = tf.function(model, input_signature).get_concrete_function().graph
options = option_builder.ProfileOptionBuilder.float_operation()

graph_info = model_analyzer.profile(forward_graph, options=options)
flops = graph_info.total_float_ops

opsDict = {}
for child in graph_info.children:
    childName: str = child.name
    childNameParts: list = childName.split("/")  ## First Part is model name
    levelName: str = childNameParts[1]

    if levelName not in opsDict:
        opsDict[levelName] = 0
    opsDict[levelName] += child.total_float_ops

for op in opsDict:
    print(f"{op} >>> {opsDict[op]}")

total = 0
for key in opsDict:
    total += opsDict[key]
print(f"TOTAL MODEL FLOATING POINT OPERATIONS >>> {flops} {total}")  ## 448572872
