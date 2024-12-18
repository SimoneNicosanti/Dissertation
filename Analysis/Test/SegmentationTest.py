import keras
import keras_hub
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Manipulation import Unnest


def plot_segmentation(original_image, preds, colormap, outPath):
    class_indices = tf.argmax(preds, axis=-1)  # Shape: (224, 224)
    segmentation_image = tf.gather(colormap, class_indices)

    plt.figure(figsize=(5, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image[0] / 255)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_image[0])
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(outPath)


def main():
    filepath = "./other/dog.png"
    image = keras.utils.load_img(filepath)
    image = np.array(image)
    image = keras.ops.expand_dims(np.array(image), axis=0)

    segmenter = keras_hub.models.DeepLabV3ImageSegmenter.from_preset(
        "deeplab_v3_plus_resnet50_pascalvoc"
    )
    preds_1 = segmenter(image, training=False)

    unnestedSegmenter = Unnest.unnestModel(segmenter)
    unnestedSegmenter.save("./models/UnnestedSegmenter.keras")
    preds_2 = unnestedSegmenter({"inputs_0": image}, training=False)[
        "segmentation_output_0"
    ]

    print(f"Norm Of Difference >>> {tf.norm(preds_1 - preds_2)}")

    cmap = plt.get_cmap("tab20").colors  # Extract colors from a predefined colormap
    colors = list(cmap[:21])

    plot_segmentation(image, preds_1, cmap, "./other/orig_segm.png")
    plot_segmentation(image, preds_2, cmap, "./other/unnest_segm.png")


def main_1():
    filepath = "./output/dog.png"
    image = keras.utils.load_img(filepath)
    image = np.array(image)
    image = keras.ops.expand_dims(np.array(image), axis=0)

    segmenter = keras_hub.models.SAMImageSegmenter.from_preset("sam_huge_sa1b")
    segmenter.summary(expand_nested=True)
    encoderLayer: keras.layers.Layer = segmenter.get_layer("sam_mask_decoder")

    print(encoderLayer._inbound_nodes[0].arguments.args)
    print(encoderLayer._inbound_nodes[0].arguments.kwargs)
    print(encoderLayer._inbound_nodes[0].arguments._flat_arguments)
    print(encoderLayer._inbound_nodes[0].arguments.keras_tensors)
    # preds_1 = segmenter(image, training=False)
    # print(preds_1.shape)
    # return

    unnestedSegmenter = Unnest.unnestModel(segmenter)
    unnestedSegmenter.save("./models/UnnestedSAM.keras")
    # return
    # preds_2 = unnestedSegmenter({"inputs_0": image}, training=False)[
    #     "segmentation_output_0"
    # ]

    # print(f"Norm Of Difference >>> {tf.norm(preds_1 - preds_2)}")

    # cmap = plt.get_cmap("tab20").colors  # Extract colors from a predefined colormap
    # colors = list(cmap[:21])

    # plot_segmentation(image, preds_1, cmap, "./output/orig_segm.png")
    # plot_segmentation(image, preds_2, cmap, "./output/unnest_segm.png")


if __name__ == "__main__":
    main()
