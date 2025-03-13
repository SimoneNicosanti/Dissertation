import cv2
import numpy
import yaml

from Client.InferenceInteractor import InferenceInteractor
from Client.PPP.YoloSegmentationPPP import YoloSegmentationPPP


def main():

    interactor = InferenceInteractor()

    classes = yaml.safe_load(open("./Client/config/coco8.yaml"))["names"]
    yolo_segmentation_ppp = YoloSegmentationPPP(640, 640, classes)

    # Pre-process
    orig_image = cv2.imread("./Client/test/Test_Image.jpg")
    preprocess_dict = yolo_segmentation_ppp.preprocess(orig_image)
    pre_image: numpy.ndarray = preprocess_dict["preprocessed_image"]

    output = interactor.start_inference("yolo11n-seg", {"images": pre_image})

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
    cv2.imwrite("./Client/test/Test_Image_Out.jpg", post_image)


if __name__ == "__main__":
    main()
