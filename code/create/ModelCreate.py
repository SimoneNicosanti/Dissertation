import keras
import numpy as np
from imageio import imread
import pickle
import tensorflow as tf

def main() -> None :
    model : keras.Model = keras.applications.MobileNetV2()
    firstLayer : keras.Layer = model.layers[1] ## Not considering Input Layer

    data = np.empty((1, 224, 224, 3))
    data[0] = imread('../data/boef.jpg')
    data = keras.applications.mobilenet_v2.preprocess_input(data)

    with open('../data/boef_pre.pkl', "wb") as f :
        pickle.dump(data, f)
    
    expArch = keras.export.ExportArchive()
    expArch.track(model)
    expArch.add_endpoint(
        name = "full",
        fn = model.__call__,
        input_signature = [tf.TensorSpec(shape=firstLayer.input.shape , dtype=tf.float32, name = "x")]
    )
    expArch.write_out(f"../../models/server/mobile_net/1")

    converter = tf.lite.TFLiteConverter.from_saved_model(f"../../models/server/mobile_net/1")
    convertedModel = converter.convert()
    with open(f"../../models/client/mobile_net.tflite", "wb") as f :
        f.write(convertedModel)




if __name__ == "__main__" :
    main()