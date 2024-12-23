import keras
import keras_cv
import keras_hub
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Manipulation import Unnester


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


def test_deep_lab():
    filepath = "./other/dog.png"
    image = keras.utils.load_img(filepath)
    image = np.array(image)
    image = keras.ops.expand_dims(np.array(image), axis=0)

    ## The keras_hub pretrained model is not working in this case!!
    ## Probably there is some custom level that is used and that is not
    ## properly exported: it may lack of the from_config method as the other
    # segmenter = keras_hub.models.DeepLabV3ImageSegmenter.from_preset("deeplab_v3_plus_resnet50_pascalvoc")

    backbone = keras_cv.models.ResNet50V2Backbone.from_preset(
        preset="resnet50_v2_imagenet",
        input_shape=(224, 224) + (3,),
        load_weights=True,
    )
    segmenter = keras_cv.models.segmentation.DeepLabV3Plus(
        num_classes=20,
        backbone=backbone,
    )

    preds_1 = segmenter(image, training=False)

    unnestedSegmenter = Unnester.unnestModel(segmenter)
    unnestedSegmenter.save("./models/UnnestedDeepLab.keras")

    loadedSegmenter = keras.saving.load_model("./models/UnnestedDeepLab.keras")
    preds_2 = loadedSegmenter({"input_layer_0": image}, training=False)[
        "sequential_7_0"
    ]

    diffNorm = tf.norm(preds_1 - preds_2)
    print(f"DeepLab Test - Norm of Diff {diffNorm}")

    cmap = plt.get_cmap("tab20").colors  # Extract colors from a predefined colormap
    colors = list(cmap[:20])
    plot_segmentation(image, preds_1, colors, "./other/orig_deep.png")
    plot_segmentation(image, preds_2, colors, "./other/unnest_deep.png")


def test_sam():
    image_size = 1024
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
    out_1 = segmenter(input_data)

    unnestedSegmenter = Unnester.unnestModel(segmenter)
    unnestedSegmenter.save("./models/UnnestedSAM.keras")

    loadedSegmenter = keras.saving.load_model("./models/UnnestedSAM.keras")

    input_data_1 = {
        "images_0": np.ones(
            (batch_size, image_size, image_size, 3),
            dtype="float32",
        ),
        "points_0": np.ones((batch_size, 1, 2), dtype="float32"),
        "labels_0": np.ones((batch_size, 1), dtype="float32"),
        "boxes_0": np.ones((batch_size, 1, 2, 2), dtype="float32"),
        "masks_0": np.zeros((batch_size, 0, image_size, image_size, 1)),
    }
    out_2 = loadedSegmenter(input_data_1)

    diffNorm = tf.norm(out_1["masks"] - out_2["sam_mask_decoder_1"])
    print(f"SAM Test - Norm of Diff {diffNorm}")


def test_seg_former():
    filepath = "./other/dog.png"
    image = keras.utils.load_img(filepath)
    image = np.array(image)
    image = keras.ops.expand_dims(np.array(image), axis=0)

    backbone = keras_cv.models.MiTBackbone.from_preset("mit_b0_imagenet")
    segmenter = keras_cv.models.segmentation.SegFormer(num_classes=1, backbone=backbone)

    out_1 = segmenter(image)

    unnestedSegmenter = Unnester.unnestModel(segmenter)
    unnestedSegmenter.save("./models/UnnestedSegformer.keras")

    loadedSegmenter = keras.saving.load_model("./models/UnnestedSegformer.keras")
    out_2 = loadedSegmenter(image)["resizing_4_0"]

    diffNorm = tf.norm(out_1 - out_2)
    print(f"SegFormer Test - Norm of Diff {diffNorm}")

    cmap = plt.get_cmap("tab20").colors  # Extract colors from a predefined colormap
    colors = list(cmap[:11])
    plot_segmentation(image, out_1, colors, "./other/orig_seg.png")
    plot_segmentation(image, out_2, colors, "./other/unnest_seg.png")


if __name__ == "__main__":
    test_deep_lab()
    test_sam()
    #test_seg_former()