import argparse
import json
import os
import time

import grpc
import numpy as np
import pandas as pd
import tqdm

from Common.ConfigReader import ConfigReader
from CommonPlan.Plan import Plan
from CommonPlan.WholePlan import WholePlan
from proto_compiled.common_pb2 import ModelId
from proto_compiled.deployment_pb2 import (
    DeploymentRequest,
    ProducePlanRequest,
    ProducePlanResponse,
)
from proto_compiled.deployment_pb2_grpc import DeploymentStub
from Test.Scripts.Utils.InferenceCaller import InferenceCaller


def get_deployer_stub():
    deployer_addr = ConfigReader("../../config/config.ini").read_str(
        "addresses", "DEPLOYER_ADDR"
    )
    deployer_port = ConfigReader("../../config/config.ini").read_int(
        "ports", "DEPLOYER_PORT"
    )
    deployer_stub = DeploymentStub(
        grpc.insecure_channel("{}:{}".format(deployer_addr, deployer_port))
    )
    return deployer_stub


def get_frontend_stub():
    frontend_addr = ConfigReader("../../config/config.ini").read_str(
        "addresses", "FRONTEND_ADDR"
    )
    frontend_port = ConfigReader("../../config/config.ini").read_int(
        "ports", "FRONTEND_PORT"
    )
    frontend_stub = DeploymentStub(
        grpc.insecure_channel("{}:{}".format(frontend_addr, frontend_port))
    )
    return frontend_stub


def build_file_name(
    case: str,
):
    return f"device_only_plan_{case}.csv"


def prepare_dataframe(dataframe: pd.DataFrame, add_df: pd.DataFrame):
    comparison_columns = [
        "model_name",
        "latency_weight",
        "energy_weight",
        "device_max_energy",
        "max_noises",
        "device_cpus",
    ]

    # Filtra righe che NON corrispondono alle tuple in add_df
    dataframe = dataframe[
        ~dataframe[comparison_columns]
        .apply(tuple, axis=1)
        .isin(add_df[comparison_columns].apply(tuple, axis=1))
    ]

    return dataframe


def generation_test(
    model_name,
    latency_weight,
    energy_weight,
    device_max_energy,
    max_noises,
    device_cpus,
    run_nums,
):
    ## Plan Production

    csv_path = os.path.join(
        "../Results/DeviceOnlyPlan",
        build_file_name(
            "Generation",
        ),
    )
    if os.path.exists(csv_path):
        dataframe = pd.read_csv(csv_path)
    else:
        dataframe = pd.DataFrame(
            columns=[
                "model_name",
                "latency_weight",
                "energy_weight",
                "device_max_energy",
                "max_noises",
                "device_cpus",
                "run_time",
                "latency_value",
                "energy_value",
                "device_energy_value",
                "quantized_layers_num",
                "quantized_layers_array",
            ]
        )

    produce_plan_request = ProducePlanRequest(
        models_ids=[ModelId(model_name=model_name)],
        latency_weight=latency_weight,
        energy_weight=energy_weight,
        device_max_energy=device_max_energy,
        requests_number=[1],
        max_noises=[max_noises],
        start_server="0",
    )

    deployer_stub = get_deployer_stub()

    produce_plan_times = np.zeros(run_nums)
    latency_values = np.zeros(run_nums)
    energy_values = np.zeros(run_nums)
    device_energy_values = np.zeros(run_nums)
    quantized_layers_num = np.zeros(run_nums)
    quantized_layers_array = []
    for idx in tqdm.tqdm(range(run_nums)):
        start = time.perf_counter_ns()
        produce_plan_response: ProducePlanResponse = deployer_stub.produce_plan(
            produce_plan_request
        )
        end = time.perf_counter_ns()
        produce_plan_times[idx] = (end - start) * 1e-9

        prod_plan: Plan = WholePlan.decode(
            json.loads(produce_plan_response.optimized_plan)
        ).get_model_plan(model_name)

        latency_values[idx] = prod_plan.get_latency_value()
        energy_values[idx] = prod_plan.get_energy_value()
        device_energy_values[idx] = (
            0 if device_max_energy is None else prod_plan.get_device_energy()
        )
        quantized_layers_num[idx] = len(prod_plan.get_quantized_nodes())

        quantized_layers = ""
        for node_id in prod_plan.get_quantized_nodes():
            quantized_layers += node_id.node_name + " "
        quantized_layers_array.append(quantized_layers)

    add_df = pd.DataFrame(
        {
            "model_name": [model_name] * run_nums,
            "latency_weight": [latency_weight] * run_nums,
            "energy_weight": [energy_weight] * run_nums,
            "device_max_energy": [device_max_energy] * run_nums,
            "max_noises": [max_noises] * run_nums,
            "device_cpus": [device_cpus] * run_nums,
            "run_time": produce_plan_times,
            "latency_value": latency_values,
            "energy_value": energy_values,
            "device_energy_value": device_energy_values,
            "quantized_layers_num": quantized_layers_num,
            "quantized_layers_array": quantized_layers_array,
        }
    )

    dataframe = prepare_dataframe(dataframe, add_df)
    dataframe = pd.concat([dataframe, add_df], ignore_index=True)
    dataframe.to_csv(csv_path, index=False)

    return produce_plan_response


def deploy_test(
    model_name,
    latency_weight,
    energy_weight,
    device_max_energy,
    max_noises,
    device_cpus,
    run_nums,
    produce_plan_response: ProducePlanResponse,
):
    csv_path = os.path.join(
        "../Results/DeviceOnlyPlan",
        build_file_name(
            "Deployment",
        ),
    )
    if os.path.exists(csv_path):
        dataframe = pd.read_csv(csv_path)
    else:
        dataframe = pd.DataFrame(
            columns=[
                "model_name",
                "latency_weight",
                "energy_weight",
                "device_max_energy",
                "max_noises",
                "device_cpus",
                "run_time",
            ]
        )

    deployer_stub = get_deployer_stub()
    deployment_request = DeploymentRequest(
        optimized_plan=produce_plan_response.optimized_plan
    )

    deploy_times = np.zeros(run_nums)
    for idx in tqdm.tqdm(range(run_nums)):
        start = time.perf_counter_ns()
        deployer_stub.deploy_plan(deployment_request)
        end = time.perf_counter_ns()

        deploy_times[idx] = (end - start) * 1e-9

    add_df = pd.DataFrame(
        {
            "model_name": [model_name] * run_nums,
            "latency_weight": [latency_weight] * run_nums,
            "energy_weight": [energy_weight] * run_nums,
            "device_max_energy": [device_max_energy] * run_nums,
            "max_noises": [max_noises] * run_nums,
            "device_cpus": [device_cpus] * run_nums,
            "run_time": deploy_times,
        }
    )

    dataframe = prepare_dataframe(dataframe, add_df)
    dataframe = pd.concat([dataframe, add_df], ignore_index=True)
    dataframe.to_csv(csv_path, index=False)

    return


def usage_test(
    model_name,
    latency_weight,
    energy_weight,
    device_max_energy,
    max_noises,
    device_cpus,
    run_nums,
):

    csv_path = os.path.join(
        "../Results/DeviceOnlyPlan",
        build_file_name(
            "Usage",
        ),
    )
    if os.path.exists(csv_path):
        dataframe = pd.read_csv(csv_path)
    else:
        dataframe = pd.DataFrame(
            columns=[
                "model_name",
                "latency_weight",
                "energy_weight",
                "device_max_energy",
                "max_noises",
                "device_cpus",
                "run_time",
            ]
        )

    ## Plan Usage
    inference_caller = InferenceCaller()

    default_rng = np.random.default_rng(seed=0)
    input_dict = {
        "images": default_rng.uniform(low=0, high=1, size=(1, 3, 640, 640)).astype(
            np.float32
        )
    }

    plan_use_times = np.zeros(run_nums)
    for idx in tqdm.tqdm(range(run_nums)):
        _, infer_time, _ = inference_caller.call_inference(model_name, input_dict)
        plan_use_times[idx] = infer_time

    add_df = pd.DataFrame(
        {
            "model_name": [model_name] * run_nums,
            "latency_weight": [latency_weight] * run_nums,
            "energy_weight": [energy_weight] * run_nums,
            "device_max_energy": [device_max_energy] * run_nums,
            "max_noises": [max_noises] * run_nums,
            "device_cpus": [device_cpus] * run_nums,
            "run_time": plan_use_times,
        }
    )

    dataframe = prepare_dataframe(dataframe, add_df)
    dataframe = pd.concat([dataframe, add_df], ignore_index=True)
    dataframe.to_csv(csv_path, index=False)

    return plan_use_times

    pass


def main():

    os.makedirs("../Results/DeviceOnlyPlan", exist_ok=True)

    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--model", type=str, help="Model Name", required=True)
    parser.add_argument(
        "--latency-weight", type=float, help="Latency Weight", required=True
    )
    parser.add_argument(
        "--energy-weight", type=float, help="Energy Weight", required=True
    )
    parser.add_argument(
        "--device-max-energy", type=float, help="Device Max Energy", default=0.0
    )
    parser.add_argument("--max-noises", type=float, help="Max Noises", default=0.0)

    parser.add_argument("--plan-gen-runs", type=int, help="Plan Gen Runs", default=10)
    parser.add_argument(
        "--plan-deploy-runs", type=int, help="Plan Deploy Runs", default=1
    )
    parser.add_argument("--plan-use-runs", type=int, help="Plan Use Runs", default=100)

    parser.add_argument("--device-cpus", help="Device CPUs", type=float, required=True)

    args = parser.parse_args()
    model_name = args.model
    latency_weight = args.latency_weight
    energy_weight = args.energy_weight
    device_max_energy = args.device_max_energy
    max_noises = args.max_noises

    plan_gen_runs = args.plan_gen_runs
    plan_deploy_runs = args.plan_deploy_runs
    plan_use_runs = args.plan_use_runs

    print("📝 Producing Plan")
    produce_plan_response: ProducePlanResponse = generation_test(
        model_name,
        latency_weight,
        energy_weight,
        device_max_energy,
        max_noises,
        args.device_cpus,
        plan_gen_runs,
    )

    print("🚀 Deploying Plan")
    deploy_test(
        model_name,
        latency_weight,
        energy_weight,
        device_max_energy,
        max_noises,
        args.device_cpus,
        plan_deploy_runs,
        produce_plan_response,
    )

    print("🤖 Using Plan")
    plan_use_times = usage_test(
        model_name,
        latency_weight,
        energy_weight,
        device_max_energy,
        max_noises,
        args.device_cpus,
        plan_use_runs,
    )
    print("🤖 Use Plan with Avg Time >> ", plan_use_times.mean())

    pass


if __name__ == "__main__":
    main()
