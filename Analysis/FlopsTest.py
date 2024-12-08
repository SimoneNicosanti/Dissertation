import csv

import keras
from Flops import ComputeFlops


def main():
    model: keras.Model = keras.saving.load_model("./models/UnpackedYolo.keras")

    # ComputeFlops.computeRunningTimes(model, (1, 64, 64, 3), 10)
    # ComputeFlops.computeFloatOperationsPerOperation(model, (1, 64, 64, 3))

    flopsPerOp: dict[str, tuple[float, float, float]] = ComputeFlops.computeFlopsPerOp(
        model, (1, 64, 64, 3), 25
    )

    with open("./models/Yolo_FLOPS.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["OpName", "FloatOps", "AvgTime", "FLOPS"])

        for opName in flopsPerOp:
            writer.writerow([opName, *flopsPerOp[opName]])


if __name__ == "__main__":
    main()
