import tensorflow as tf
from tensorflow.python.profiler import model_analyzer, option_builder
import keras

model : keras.Model = keras.applications.MobileNetV2()

input_signature = [
    tf.TensorSpec(
        shape=(1, *params.shape[1:]), 
        dtype=params.dtype, 
        name=params.name
    ) for params in model.inputs
]

forward_graph = tf.function(model, input_signature).get_concrete_function().graph
options = option_builder.ProfileOptionBuilder.float_operation()
graph_info = model_analyzer.profile(forward_graph, options=options)
flops = graph_info.total_float_ops
print(f"TOTAL MODEL FLOPS >>> {flops}") ## 615292208