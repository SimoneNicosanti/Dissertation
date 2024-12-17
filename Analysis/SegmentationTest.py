import keras
import keras_cv
import keras_hub
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import ops
from keras.preprocessing.image import img_to_array, load_img
from Manipulation import Unnest
from PIL import Image

images = np.ones(shape=(1, 96, 96, 3))
labels = np.zeros(shape=(1, 96, 96, 2))
image_converter = keras_hub.layers.DeepLabV3ImageConverter(
    image_size=(512, 512),
    interpolation="bilinear",
)
preprocessor = keras_hub.models.DeepLabV3ImageSegmenterPreprocessor(image_converter)
segmenter = keras_hub.models.DeepLabV3ImageSegmenter.from_preset(
    "deeplab_v3_plus_resnet50_pascalvoc",
)
segmenter.preprocessor = preprocessor
segmenter.summary()

# filepath = keras.utils.get_file(
#     origin="https://storage.googleapis.com/keras-cv/models/paligemma/cow_beach_1.png"
# )
# filepath = "./shih-tzu.png"
# image = keras.utils.load_img(filepath)
# image = np.array(image)
# model = keras_hub.models.DeepLabV3ImageSegmenter.from_preset(
#     "deeplab_v3_plus_resnet50_pascalvoc"
# )

# image = preprocessor(image)
# image = keras.ops.expand_dims(np.array(image), axis=0)
# preds = ops.expand_dims(ops.argmax(model.predict(image), axis=-1), axis=-1)


def plot_segmentation(original_image, predicted_mask):
    plt.figure(figsize=(5, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image[0] / 255)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask[0])
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("segmentation_output.png")


plot_segmentation(image, preds)
