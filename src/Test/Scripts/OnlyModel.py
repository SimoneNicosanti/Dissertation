import argparse
import os
import subprocess
import time
from ast import arg

import numpy as np
import onnxruntime
import pandas as pd
import tqdm


def run_onnx_model(input: np.ndarray, model_name: str, runs: int):

    model_path = "/model_pool_data/models/" + model_name + ".onnx"

    device_type = "cpu"
    providers = []
    sess_options = onnxruntime.SessionOptions()
    if (
        "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        and subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.DEVNULL).returncode
        == 0
    ):
        device_type = "cuda"
        providers.append("CUDAExecutionProvider")
    elif "OpenVINOExecutionProvider" in onnxruntime.get_available_providers():
        providers.append("OpenVINOExecutionProvider")
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
    else:
        providers.append("CPUExecutionProvider")

    sess = onnxruntime.InferenceSession(
        model_path,
        providers=providers,
        sess_options=sess_options,
    )

    input_name = sess.get_inputs()[0].name

    input_dict = {
        input_name: onnxruntime.OrtValue.ortvalue_from_numpy(
            input, device_type=device_type
        )
    }
    ## Cold Start
    for _ in range(10):
        sess.run_with_ort_values(None, input_dict)

    time_array = np.zeros(runs)

    for i in tqdm.tqdm(range(0, runs)):
        start = time.perf_counter_ns()
        sess.run_with_ort_values(None, input_dict)
        end = time.perf_counter_ns()
        time_array[i] = (end - start) * 1e-9

    return time_array


def main():

    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument(
        "--models", nargs="+", type=str, help="Model Names", required=True
    )
    parser.add_argument("--cpus", type=float, help="Number of CPUs", required=True)
    parser.add_argument("--runs", type=int, help="Number of Runs", default=50)
    parser.add_argument("--server", type=str, help="Server Name", required=True)

    args = parser.parse_args()
    model_names = args.models
    cpus = args.cpus
    runs = args.runs

    if os.path.exists(
        f"/src/Test/Results/DeviceOnlyModel/{args.server}_only_model.csv"
    ):
        dataframe = pd.read_csv(
            f"/src/Test/Results/DeviceOnlyModel/{args.server}_only_model.csv"
        )
    else:
        dataframe = pd.DataFrame(columns=["model_name", "cpus", "profile_time"])
        os.makedirs("/src/Test/Results/DeviceOnlyModel", exist_ok=True)

    generator = np.random.default_rng(seed=0)
    input = generator.uniform(low=0, high=1, size=(1, 3, 640, 640))
    input = input.astype(np.float32)
    for model_name in model_names:
        print("Processing >> ", model_name)
        time_array = run_onnx_model(input, model_name, runs)
        print(np.mean(time_array))

        new_df = pd.DataFrame(
            {
                "model_name": [model_name] * runs,
                "cpus": [cpus] * runs,
                "run_time": time_array,
            }
        )

        dataframe = dataframe[
            ~((dataframe["model_name"] == model_name) & (dataframe["cpus"] == cpus))
        ]
        dataframe = pd.concat([dataframe, new_df], ignore_index=True)

    dataframe.to_csv(
        f"/src/Test/Results/DeviceOnlyModel/{args.server}_only_model.csv", index=False
    )

    pass


if __name__ == "__main__":
    main()
