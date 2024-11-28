import time

import keras
import numpy as np


class LayerExecutionTime(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.layer_times: dict[str, list] = {}
        self.start_time = 0
        self.totals = []

    def on_predict_batch_begin(self, batch, logs=None):
        self.start_time = time.time_ns()

    def on_predict_batch_end(self, batch, logs=None):
        self.totals.append(time.time_ns() - self.start_time)

    def wrap_layer(self, layer):
        original_call = layer.call

        def timed_call(inputs, *args, **kwargs):
            start_time = time.time_ns()
            result = original_call(inputs, *args, **kwargs)
            layer_name = layer.name
            duration = time.time_ns() - start_time
            if layer_name not in self.layer_times:
                self.layer_times[layer_name] = []
            self.layer_times[layer_name].append(duration)
            return result

        layer.call = timed_call

    def set_model(self, model):
        super().set_model(model)
        for layer in model.layers:
            self.wrap_layer(layer)


# Example usage
model = keras.applications.MobileNetV3Large()

x = np.random.rand(1, 32, 32, 3)

callback = LayerExecutionTime()
for _ in range(0, 100):
    model.predict(x, callbacks=[callback])

# Access layer timings
print("\nLayer-wise Execution Times:")
for layer, times in callback.layer_times.items():
    avgLayerTime = sum(times) / len(times)
    print(f"Layer {layer}: Avg Time = {avgLayerTime:.4f} nano sec")
avgTotal = sum(callback.totals) / len(callback.totals)
print(f"Avg Total >>> {avgTotal:.4f} nano sec")
