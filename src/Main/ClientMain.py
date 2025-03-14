import threading

import cv2
import numpy
import yaml

from Client.InferenceCaller import InferenceCaller
from Client.PPP.YoloPPP import YoloPPP
from Client.PPP.YoloSegmentationPPP import YoloSegmentationPPP


def main():

    interactor = InferenceCaller()

    classes = yaml.safe_load(open("./Client/config/coco8.yaml"))["names"]
    yolo_segmentation_ppp = YoloSegmentationPPP(640, 640, classes)

    # Pre-process
    orig_image = cv2.imread("./Client/test/Test_Image.jpg")
    preprocess_dict = yolo_segmentation_ppp.preprocess(orig_image)
    pre_image: numpy.ndarray = preprocess_dict["preprocessed_image"]

    model_list = [
        "yolo11n-seg",
    ]
    thr_list = []
    for idx in range(18):
        thr = threading.Thread(
            target=do_inference,
            args=(
                interactor,
                yolo_segmentation_ppp,
                preprocess_dict,
                pre_image,
                orig_image,
                model_list[idx % len(model_list)],
            ),
        )
        thr_list.append(thr)

    for thr in thr_list:
        thr.start()

    for thr in thr_list:
        thr.join()


def do_inference(
    inference_caller: InferenceCaller,
    yolo_segmentation_ppp: YoloPPP,
    preprocess_dict: dict[str],
    pre_image,
    orig_image,
    model_name,
):
    output, request_idx = inference_caller.call_inference(
        model_name, {"images": pre_image}
    )

    output0 = output["output0"]
    output1 = output["output1"]

    post_image = yolo_segmentation_ppp.postprocess(
        orig_image,
        [output0, output1],
        0.5,
        0.5,
        ratio=preprocess_dict["ratio"],
        pad_w=preprocess_dict["pad_w"],
        pad_h=preprocess_dict["pad_h"],
        nm=32,
    )
    cv2.imwrite(f"./Client/test/Test_Image_Out_{request_idx}.jpg", post_image)


if __name__ == "__main__":
    main()
