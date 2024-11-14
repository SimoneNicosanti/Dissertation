import tensorflow as tf
import keras
import functools
import numpy as np
from imageio import imread
import pickle
from keras.applications.mobilenet_v2 import preprocess_input

BLOCK_SIZE = 10









def main():
    model : keras.Model = keras.applications.MobileNetV2()
    firstLayer : keras.Layer = model.layers[1] ## Not considering Input Layer

    data = np.empty((1, 224, 224, 3))
    data[0] = imread('../client/boef.jpg')
    data = preprocess_input(data)

    with open('../client/boef_pre.pkl', "wb") as f :
        pickle.dump(data, f)
    


    expArch = keras.export.ExportArchive()
    expArch.track(model)
    expArch.add_endpoint(
        name = "full",
        fn = model.__call__,
        input_signature = [tf.TensorSpec(shape=firstLayer.input.shape , dtype=tf.float32, name = "x")]
    )
    expArch.write_out(f"../server/models/mobile_net/1")

    converter = tf.lite.TFLiteConverter.from_saved_model(f"../server/models/mobile_net/1")
    convertedModel = converter.convert()
    with open(f"../client/models/mobile_net.tflite", "wb") as f :
        f.write(convertedModel)



if __name__ == "__main__":
    main()
