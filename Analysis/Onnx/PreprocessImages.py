import os

import cv2
import numpy as np
from YoloPPP import YoloPPP

path = (
    "../../coco128/images/train2017"  # Sostituisci con il percorso della tua directory
)
out_path = "../../coco128/"
files = os.listdir(path)

# Elenca solo i file (non le directory)
files = [f for f in files if os.path.isfile(os.path.join(path, f))]
# print(files)
yolo_ppp = YoloPPP(640, 640)

all_images = []
for file_name in files:
    print(file_name)
    file_path = os.path.join(path, file_name)
    prep_image = yolo_ppp.preprocess(cv2.imread(file_path))

    all_images.append(prep_image[0])

all_images = np.asarray(all_images, dtype=np.float32)
print(all_images.shape)
np.save(os.path.join(out_path, "all_preps.npy"), all_images)

loaded = np.load(os.path.join(out_path, "all_preps.npy"))

print(loaded.shape)
