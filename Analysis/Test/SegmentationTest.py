import keras
import keras_cv
import keras_hub
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Manipulation import Unnest, Unnester, Utils


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
    segmenter.summary(expand_nested=True)
    print(segmenter.get_layer("inputs")._outbound_nodes)
    print(Utils.getInputLayersNames(segmenter.get_layer("deep_lab_v3_backbone")))


    preds_1 = segmenter(image, training=False)

    unnestedSegmenter = Unnester.unnestModel(segmenter)
    unnestedSegmenter.save("./models/UnnestedSegmenter.keras")
    preds_2 = unnestedSegmenter({"inputs_0": image}, training=False)[
        "segmentation_output_0"
    ]

    print(f"Norm Of Difference >>> {tf.norm(preds_1 - preds_2)}")

    cmap = plt.get_cmap("tab20").colors  # Extract colors from a predefined colormap
    colors = list(cmap[:21])
    plot_segmentation(image, preds_1, colors, "./other/orig_segm.png")
    plot_segmentation(image, preds_2, colors, "./other/unnest_segm.png")


def main_1():
    image_size = 128
    batch_size = 2
    input_data = {
        "images": np.ones(
            (batch_size, image_size, image_size, 3),
            dtype="float32",
        ),
        "points": np.ones((batch_size, 1, 2), dtype="float32"),
        "labels": np.ones((batch_size, 1), dtype="float32"),
        "boxes": np.ones((batch_size, 1, 2, 2), dtype="float32"),
        "masks": np.zeros((batch_size, 0, image_size, image_size, 1)),
    }
    segmenter = keras_hub.models.SAMImageSegmenter.from_preset("sam_base_sa1b")
    segmenter.summary()
    segmenter.summary(expand_nested=True)
    print(segmenter.get_layer("sam_backbone").get_layer("images").output._keras_history)
    print(segmenter.get_layer("images").output._keras_history)
    print(segmenter.get_layer("images")._inbound_nodes)
    # seq = segmenter.get_layer("sam_backbone").get_layer("sequential_12")
    # print(segmenter.get_layer("boxes")._outbound_nodes)
    # outputs = segmenter.predict(input_data)
    # print(outputs)
    # preds_1 = segmenter(image, training=False)
    # print(preds_1.shape)
    # return

    unnestedSegmenter = Unnester.unnestModel(segmenter)
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


def main_2():
    filepath = "./other/dog.png"
    image = keras.utils.load_img(filepath)
    image = np.array(image)
    image = keras.ops.expand_dims(np.array(image), axis=0)

    segmenter = keras_cv.models.SegFormer.from_preset("segformer_b5", num_classes=10)
    out_1 = segmenter(image)

    unnestedSegmenter = Unnest.unnestModel(segmenter)
    unnestedSegmenter.save("./models/UnnestedSegformer.keras")
    out_2 = unnestedSegmenter(image)["resizing_4_0"]

    print("Are Equal >> ", np.array_equal(out_1, out_2))

    cmap = plt.get_cmap("tab20").colors  # Extract colors from a predefined colormap
    colors = list(cmap[:11])
    plot_segmentation(image, out_1, colors, "./other/orig_segm.png")
    plot_segmentation(image, out_2, colors, "./other/unnest_segm.png")

if __name__ == "__main__":
    main_1()