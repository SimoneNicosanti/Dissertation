import argparse
import os
import time

import grpc

from Common import ConfigReader
from proto_compiled.common_pb2 import ModelId
from proto_compiled.server_pb2 import ExecutionProfileRequest, ExecutionProfileResponse
from proto_compiled.server_pb2_grpc import ExecutionProfileStub
import pandas as pd
import numpy as np

SERVER_MAP = {
    "Client": "10.0.1.15",
    "Edge": "10.0.1.16",
    "Cloud": "10.0.1.17"
}


def get_profiler_stub(machine_name: str):
    ip_addr = SERVER_MAP[machine_name]
    port = ConfigReader.ConfigReader("../../config/config.ini").read_int(
        "ports", "EXECUTION_PROFILER_PORT"
    )
    profiler_stub = ExecutionProfileStub(grpc.insecure_channel(f"{ip_addr}:{port}"))

    return profiler_stub


def main():

    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--model", type=str, help="Model Name", required=True)
    parser.add_argument("--cpus", type=float, help="Number of CPUs", required=True)
    parser.add_argument("--server", type=str, help="Server Name", required=True)
    parser.add_argument("--runs", type=int, help="Number of Runs", default=1)
    
    args = parser.parse_args()

    model_names = args.model
    cpus = args.cpus
    server_name = args.server
    runs = args.runs


    csv_file_name = f"/src/Test/Results/Exec_Profile_Time/{server_name}_exec_profile_time.csv"
    directory = os.path.dirname(csv_file_name)
    os.makedirs(directory, exist_ok=True)
    
    if os.path.exists(csv_file_name):
        dataframe = pd.read_csv(csv_file_name)
    else:
        dataframe = pd.DataFrame(columns=["model_name", "cpus", "profile_time"])

    profiler_stub = get_profiler_stub(server_name)

    for model_name in model_names:
        print("Running for Model >> ", model_name)
        time_array = np.zeros(runs)
        for idx in range(runs):
            print("\t Run Number >> ", idx)
            execution_profile_request = ExecutionProfileRequest(
                model_id=ModelId(model_name=model_name)
            )

            start = time.perf_counter_ns()
            exec_profile_response: ExecutionProfileResponse = profiler_stub.profile_execution(
                execution_profile_request
            )
            end = time.perf_counter_ns()

            time_array[idx] = (end - start) * 1e-9
        
        add_df = pd.DataFrame(
            {
                "model_name": [model_name] * runs,
                "cpus": [cpus] * runs,
                "profile_time": time_array,
            }
        )

        dataframe = dataframe[~((dataframe["model_name"] == model_name) & (dataframe["cpus"] == cpus))]
        dataframe = pd.concat([dataframe, add_df], ignore_index=True)

        exec_profile_json = exec_profile_response.profile

        profile_file_name = f"/src/Test/Results/Exec_Profile/{server_name}_{model_name}_{cpus}_exec_profile.json"
        directory = os.path.dirname(profile_file_name)
        os.makedirs(directory, exist_ok=True)

        with open(profile_file_name, "w") as f:
            f.write(exec_profile_json)
    
    dataframe.to_csv(csv_file_name, index=False)

    


if __name__ == "__main__":
    main()
