import argparse
import json
import time

import grpc
import numpy as np
import pandas as pd

from Common.ConfigReader import ConfigReader
from CommonProfile.ModelProfile import ModelProfile
from proto_compiled.common_pb2 import ModelId
from proto_compiled.model_profile_pb2 import ProfileRequest, ProfileResponse
from proto_compiled.model_profile_pb2_grpc import ModelProfileStub

MODEL_NAME = "yolo11"
SIZE_LIST = ["n", "s", "m", "l", "x"]
CASE_LIST = ["", "-seg", "-cls"]


def build_profile_list(model_names: str):
    if model_names is None:
        model_names = []
        for size in SIZE_LIST:
            for case in CASE_LIST:
                model_names.append(MODEL_NAME + size + case)

    return model_names


def get_profiler_stub():
    ip_addr = ConfigReader("../../config/config.ini").read_str(
        "addresses", "MODEL_PROFILER_ADDR"
    )
    port = ConfigReader("../../config/config.ini").read_int(
        "ports", "MODEL_PROFILER_PORT"
    )
    model_pool_chann = grpc.insecure_channel("{}:{}".format(ip_addr, port))

    model_profiler = ModelProfileStub(model_pool_chann)
    return model_profiler


def profile_nodes_edges(model_name: str):
    model_profiler = get_profiler_stub()

    dataframe = pd.DataFrame(columns=["model_case", "num_nodes", "num_edges"])

    for mod_size in SIZE_LIST:
        for mod_case in CASE_LIST:
            final_model_name = MODEL_NAME + mod_size + mod_case

            profile_request = ProfileRequest(
                model_id=ModelId(model_name=final_model_name), profile_regression=False
            )

            time.perf_counter_ns()
            profile_response: ProfileResponse = model_profiler.profile_model(
                profile_request
            )
            time.perf_counter_ns()

            # print("Profile Time >> ", (end - start) * 1e-9)

            model_profile: ModelProfile = ModelProfile.decode(
                json.loads(profile_response.model_profile)
            )

            num_nodes = len(model_profile.get_model_graph().nodes)
            num_edges = len(model_profile.get_model_graph().edges)

            model_case = mod_size + (mod_case if mod_case != "" else "-det")
            new_df = pd.DataFrame(
                {
                    "model_case": [model_case],
                    "num_nodes": [num_nodes],
                    "num_edges": [num_edges],
                }
            )

            dataframe = pd.concat([dataframe, new_df], ignore_index=True)

    dataframe.to_csv("/src/Test/Results/model_sizes.csv", index=False)

    pass


def whole_profile(model_names: str, runs: int):
    model_names = build_profile_list(model_names)
    print(model_names)

    model_profiler = get_profiler_stub()

    for curr_model_name in model_names:
        dataframe = pd.DataFrame(columns=["model_name", "profile_time"])
        time_array = np.zeros(runs)

        for idx in range(runs):
            profile_request = ProfileRequest(
                model_id=ModelId(model_name=curr_model_name), profile_regression=True
            )

            start = time.perf_counter_ns()
            model_profile: ProfileResponse = model_profiler.profile_model(
                profile_request
            )
            end = time.perf_counter_ns()

            time_array[idx] = (end - start) * 1e-9

        add_df = pd.DataFrame(
            {
                "model_name": [curr_model_name] * runs,
                "profile_time": time_array,
            }
        )

        dataframe = pd.concat([dataframe, add_df], ignore_index=True)
        dataframe.to_csv(
            f"/src/Test/Results/Model_Profile_Time/{curr_model_name}_profile_time.csv",
            index=False,
        )

        model_profile_json = model_profile.model_profile
        with open(f"/src/Test/Results/Model_Profile/{curr_model_name}.json", "w") as f:
            f.write(model_profile_json)


def main():

    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument(
        "--model", type=str, help="Model Name", nargs="+", required=True
    )
    parser.add_argument(
        "--case", type=str, help="Case", default="", choices=["size", "whole"]
    )
    parser.add_argument("--runs", type=int, help="Number of Runs", default=1)

    args = parser.parse_args()
    model_names = args.model
    prof_case = args.case
    runs = args.runs

    if model_names[0] == "ALL" and prof_case == "size":
        profile_nodes_edges(None)
    elif model_names[0] == "ALL" and prof_case == "whole":
        whole_profile(None, runs)
    elif model_names[0] != "ALL" and prof_case == "size":
        profile_nodes_edges(model_names)
    elif model_names[0] != "ALL" and prof_case == "whole":
        whole_profile(model_names, runs)


if __name__ == "__main__":
    main()
