import time

import cv2
import numpy as np
import scipy.stats as stats
import supervision as sv

from Client.InferenceCaller import InferenceCaller
from Client.YoloPPP import YoloPPP


def main():

    interactor = InferenceCaller()

    yolo_ppp = YoloPPP(640, 640)

    # Pre-process
    orig_image = cv2.imread("./Client/test/Test_Image.jpg")
    pre_image = yolo_ppp.preprocess(orig_image)

    times = np.zeros(100)
    for idx in range(100):
        start = time.perf_counter_ns()
        output0, output1 = do_inference(
            interactor,
            pre_image,
            "yolo11x-seg",
        )
        end = time.perf_counter_ns()
        times[idx] = end - start

    times = times * 1e-9
    mean = times.mean()
    std_err = stats.sem(times)  # errore standard
    print("Avg Inference Time >>> ", times.mean())

    conf_level = 0.95
    ci = stats.t.interval(conf_level, df=len(times) - 1, loc=mean, scale=std_err)
    print(f"Media: {mean:.3f}")
    print(f"Intervallo di confidenza al 95%: ({ci[0]:.3f}, {ci[1]:.3f})")


# # Post-process
# bboxes, masks, _ = yolo_ppp.postprocess(
#     orig_image,
#     predictions=output0,
#     prototypes=output1,
#     score_thr=0.5,
#     iou_thr=0.5,
#     num_classes=80,
# )

# if bboxes is not None and masks is not None:

#     detections = sv.Detections(
#         xyxy=bboxes[:, :4],
#         mask=masks,
#         confidence=bboxes[:, 4],
#         class_id=bboxes[:, 5].astype(int),
#     )

#     mask_annotator = sv.MaskAnnotator()

#     # Applica le annotazioni a un'immagine
#     annotated_image = mask_annotator.annotate(
#         scene=orig_image,  # la tua immagine (array NumPy)
#         detections=detections,  # le detections da disegnare
#     )

#     labels = [
#         f"{class_name} {confidence:.2f}"
#         for class_name, confidence in zip(
#             detections.class_id, detections.confidence
#         )
#     ]

#     box_annotator = sv.BoxAnnotator()

#     # Applica le annotazioni a un'immagine
#     annotated_image = box_annotator.annotate(
#         scene=orig_image,  # la tua immagine (array NumPy)
#         detections=detections,  # le detections da disegnare
#     )

#     label_annotator = sv.LabelAnnotator()

#     # Applica le annotazioni a un'immagine
#     annotated_image = label_annotator.annotate(
#         scene=orig_image,  # la tua immagine (array NumPy)
#         detections=detections,  # le detections da disegnare
#         labels=labels,
#     )

#     cv2.imwrite("./Client/test/Test_Image_Out.jpg", annotated_image)


def do_inference(
    inference_caller: InferenceCaller,
    pre_image: np.ndarray,
    model_name,
):
    output, request_idx = inference_caller.call_inference(
        model_name, {"images": pre_image}
    )

    output0 = output["output0"]
    output1 = output["output1"]

    return output0, output1


if __name__ == "__main__":
    main()
