import os

import cv2
import numpy as np
from YoloSegmentationPPP import YoloSegmentationPPP

path = (
    "../../coco128/images/train2017"  # Sostituisci con il percorso della tua directory
)
out_path = "../../coco128/preprocessed"
files = os.listdir(path)

# Elenca solo i file (non le directory)
files = [f for f in files if os.path.isfile(os.path.join(path, f))]
# print(files)
yolo_segmentation_ppp = YoloSegmentationPPP(640, 640)

for file_name in files:
    print(file_name)
    file_path = os.path.join(path, file_name)
    prep_dict = yolo_segmentation_ppp.preprocess(cv2.imread(file_path))
    prep_image = prep_dict["preprocessed_image"]

    np.savez(os.path.join(out_path, file_name.replace(".jpg", ".npz")), prep_image)
