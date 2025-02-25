import keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# Load or build your Keras model
model = keras.applications.ResNet50()
# Alternatively, define your model:
# model = tf.keras.Sequential([...])

# Apply quantization aware training
# This wraps your model with quantization operations.
quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)

# Compile the quantization-aware model
q_aware_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)
q_aware_model.save("./models/quantization_aware_model.keras")

# Fine-tune the model for a few epochs (adjust epochs as needed)
q_aware_model.fit(
    train_data, train_labels, epochs=5, validation_data=(val_data, val_labels)
)

# Evaluate the quantization-aware model
loss, accuracy = q_aware_model.evaluate(test_data, test_labels)
print("Quantization Aware Model Accuracy:", accuracy)

# Save the quantized model in the standard TensorFlow/Keras format
q_aware_model.save("quantization_aware_model.h5")
