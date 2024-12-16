from concurrent import futures

import grpc
import keras
import tensorflow as tf
from Manipulation import Split
from proto import registry_pb2_grpc
from Registry import Registry


def prepareSubModel(model: keras.Model):
    subModels: list[keras.Model] = Split.modelSplit(model, 50)
    for i, mod in enumerate(subModels):
        mod.save(f"/models/SubModel_{i}.keras")

        archive = keras.export.ExportArchive()
        archive.track(mod)

        inputs = {}
        for key in mod.input:
            inpTens = mod.input[key]
            inputs[key] = tf.TensorSpec(
                shape=inpTens.shape, dtype=inpTens.dtype, name=inpTens.name
            )
        archive.add_endpoint(
            name="serve",
            fn=mod.call,
            input_signature=[inputs],
        )
        archive.write_out(f"/tmp/SubModel_{i}")

        converter = tf.lite.TFLiteConverter.from_saved_model(f"/tmp/SubModel_{i}")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Operazioni TFLite
            tf.lite.OpsSet.SELECT_TF_OPS,  # Fallback ai kernel TensorFlow
        ]
        tflite_model = converter.convert()
        with open(f"/models/SubModel_{i}.tflite", "wb") as f:
            f.write(tflite_model)

    return subModels


def main():
    # model: keras.Model = keras.applications.MobileNetV3Large()
    model: keras.Model = keras.saving.load_model("/models/UnnestedYolo.keras")

    subModels = prepareSubModel(model)

    print("Model Parts >>> ", len(subModels))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    registry_pb2_grpc.add_RegisterServicer_to_server(
        Registry(partsNum=len(subModels), outputsNames=list(model.output.keys())),
        server,
    )
    server.add_insecure_port("[::]:5000")
    server.start()
    print("Registry Started")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
