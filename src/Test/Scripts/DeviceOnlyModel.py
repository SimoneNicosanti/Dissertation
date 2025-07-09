import argparse
import os
import time

import numpy as np
import onnxruntime
import pandas as pd
import tqdm


def run_onnx_model(input: np.ndarray, model_name: str, runs: int):

    model_path = "/model_pool_data/models/" + model_name + ".onnx"

    providers = []
    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        providers.append("CUDAExecutionProvider")
    elif "OpenVINOExecutionProvider" in onnxruntime.get_available_providers():
        providers.append("OpenVINOExecutionProvider")
    else:
        providers.append("CPUExecutionProvider")

    sess = onnxruntime.InferenceSession(model_path, providers=providers)

    input_name = sess.get_inputs()[0].name

    ## Cold Start
    sess.run(None, {input_name: input})

    time_array = np.zeros(runs)

    for i in tqdm.tqdm(range(0, runs)):
        start = time.perf_counter_ns()
        sess.run(None, {input_name: input})
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
    parser.add_argument("--runs", type=int, help="Number of Runs", default=1)

    args = parser.parse_args()
    model_names = args.models
    cpus = args.cpus
    runs = args.runs

    if os.path.exists("/src/Test/Results/DeviceOnlyModel/device_only_model.csv"):
        dataframe = pd.read_csv(
            "/src/Test/Results/DeviceOnlyModel/device_only_model.csv"
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
        "/src/Test/Results/DeviceOnlyModel/device_only_model.csv", index=False
    )

    pass


if __name__ == "__main__":
    main()
