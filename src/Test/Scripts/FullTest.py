


import argparse
import json
import os
import re
import time

import grpc
import numpy as np
import pandas as pd
import tqdm

from Common.ConfigReader import ConfigReader
from CommonIds.NodeId import NodeId
from CommonPlan.Plan import Plan
from CommonPlan.WholePlan import WholePlan
from Test.Scripts.Utils.InferenceCaller import InferenceCaller
from proto_compiled.common_pb2 import ModelId
from proto_compiled.deployment_pb2 import DeploymentRequest, ProducePlanRequest, ProducePlanResponse
from proto_compiled.deployment_pb2_grpc import DeploymentStub

def main():

    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--model-name", type=str, help="Model Name", required=True)

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

    ## Device Always Present
    parser.add_argument("--device-cpus", help="Device CPUs", type=float, required=True)

    parser.add_argument("--edge", action="store_true", help="Activate Edge")
    parser.add_argument("--cloud", action="store_true", help="Activate Cloud")

    parser.add_argument("--edge-cpus", help="Edge CPUs", type=float)
    parser.add_argument("--cloud-cpus", help="Cloud CPUs", type=float)

    parser.add_argument(
        "--device-bandwidth", help="Device Bandwidth", type=float
    )
    parser.add_argument(
        "--edge-bandwidth", help="Edge Bandwidth", type=float
    )
    parser.add_argument(
        "--cloud-bandwidth", help="Cloud Bandwidth", type=float
    )

    args = parser.parse_args()

    plan_gen_runs = args.plan_gen_runs
    plan_deploy_runs = args.plan_deploy_runs
    plan_use_runs = args.plan_use_runs

    device_bandwidth = args.device_bandwidth
    
    has_edge = args.edge
    edge_cpus = args.edge_cpus
    edge_bandwidth = args.edge_bandwidth
    if has_edge and (edge_cpus is None or edge_bandwidth is None or device_bandwidth is None):
        raise ValueError("Edge CPUs and Edge/Device Bandwidth must be specified if Edge is active")
    
    has_cloud = args.cloud
    cloud_cpus = args.cloud_cpus
    cloud_bandwidth = args.cloud_bandwidth
    if has_cloud and (cloud_cpus is None or cloud_bandwidth is None or device_bandwidth is None):
        raise ValueError("Cloud CPUs and Cloud/Device Bandwidth must be specified if Cloud is active")


    result_folder = build_result_folder(has_edge, has_cloud)

    print("📝 Producing Plan")
    produce_plan_response: ProducePlanResponse = generation_test(args, result_folder, plan_gen_runs)

    print("🚀 Deploying Plan")
    deployment_test(args, result_folder, plan_deploy_runs, produce_plan_response)

    print("🤖 Using Plan")
    plan_use_times : np.ndarray = usage_test(args, result_folder, plan_use_runs)
    print("🤖 Use Plan with Avg Time >> ", plan_use_times.mean())


def generation_test(args : argparse.Namespace, result_folder : str, run_nums : int) :
    csv_path = os.path.join(
        result_folder, "generation.csv"
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
                "edge_cpus",
                "cloud_cpus",
                "device_bandwidth",
                "edge_bandwidth",
                "cloud_bandwidth",
                "run_time",
                "latency_value",
                "energy_value",
                "device_energy_value",
                "quantized_layers_num",
                "quantized_layers_array",
                "device_components",
                "edge_components",
                "cloud_components",
            ]
        )
    
    produce_plan_request = ProducePlanRequest(
        models_ids=[ModelId(model_name=args.model_name)],
        latency_weight=args.latency_weight,
        energy_weight=args.energy_weight,
        device_max_energy=args.device_max_energy,
        requests_number=[1],
        max_noises=[args.max_noises],
        start_server="0",
    )

    deployer_stub = get_deployer_stub()

    produce_plan_times = np.zeros(run_nums)
    latency_values = np.zeros(run_nums)
    energy_values = np.zeros(run_nums)
    device_energy_values = np.zeros(run_nums)
    quantized_layers_num = np.zeros(run_nums)
    quantized_layers_array = []
    device_comps_array = []
    edge_comps_array = []
    cloud_comps_array = []
    for idx in tqdm.tqdm(range(run_nums)):
        start = time.perf_counter_ns()
        produce_plan_response: ProducePlanResponse = deployer_stub.produce_plan(
            produce_plan_request
        )
        end = time.perf_counter_ns()
        produce_plan_times[idx] = (end - start) * 1e-9

        if produce_plan_response.optimized_plan == "":
            continue

        prod_plan: Plan = WholePlan.decode(
            json.loads(produce_plan_response.optimized_plan)
        ).get_model_plan(args.model_name)

        latency_values[idx] = prod_plan.get_latency_value()
        energy_values[idx] = prod_plan.get_energy_value()
        device_energy_values[idx] = (
            0 if args.device_max_energy is None else prod_plan.get_device_energy()
        )
        quantized_layers_num[idx] = len(prod_plan.get_quantized_nodes())

        quantized_layers = ""
        for node_id in prod_plan.get_quantized_nodes():
            quantized_layers += node_id.node_name + " "
        quantized_layers_array.append(quantized_layers)

        device_comps_num = 0
        edge_comps_num = 0
        cloud_comps_num = 0
        for comp_id in prod_plan.get_all_components():
            if comp_id.net_node_id == NodeId("0"):
                device_comps_num += 1
            elif comp_id.net_node_id == NodeId("1"):
                edge_comps_num += 1
            else:
                cloud_comps_num += 1
        device_comps_array.append(device_comps_num)
        edge_comps_array.append(edge_comps_num)
        cloud_comps_array.append(cloud_comps_num)

    add_df = pd.DataFrame(
        {
            "model_name": [args.model_name] * run_nums,
            "latency_weight": [args.latency_weight] * run_nums,
            "energy_weight": [args.energy_weight] * run_nums,
            "device_max_energy": [args.device_max_energy] * run_nums,
            "max_noises": [args.max_noises] * run_nums,
            "device_cpus": [args.device_cpus] * run_nums,
            "edge_cpus": [args.edge_cpus] * run_nums,
            "cloud_cpus": [args.cloud_cpus] * run_nums,
            "device_bandwidth": [args.device_bandwidth] * run_nums,
            "edge_bandwidth": [args.edge_bandwidth] * run_nums,
            "cloud_bandwidth": [args.cloud_bandwidth] * run_nums,
            "run_time": produce_plan_times,
            "latency_value": latency_values,
            "energy_value": energy_values,
            "device_energy_value": device_energy_values,
            "quantized_layers_num": quantized_layers_num,
            "quantized_layers_array": quantized_layers_array,
            "device_components": device_comps_array,
            "edge_components": edge_comps_array,
            "cloud_components": cloud_comps_array,
        }
    )

    dataframe = prepare_dataframe(dataframe, add_df)
    dataframe = pd.concat([dataframe, add_df], ignore_index=True)
    dataframe.to_csv(csv_path, index=False)

    return produce_plan_response

def deployment_test(args : argparse.Namespace, result_folder : str, run_nums : int, produce_plan_response: ProducePlanResponse):
    csv_path = os.path.join(
        result_folder, "deployment.csv"
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
                "edge_cpus",
                "cloud_cpus",
                "device_bandwidth",
                "edge_bandwidth",
                "cloud_bandwidth",
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
            "model_name": [args.model_name] * run_nums,
            "latency_weight": [args.latency_weight] * run_nums,
            "energy_weight": [args.energy_weight] * run_nums,
            "device_max_energy": [args.device_max_energy] * run_nums,
            "max_noises": [args.max_noises] * run_nums,
            "device_cpus": [args.device_cpus] * run_nums,
            "edge_cpus": [args.edge_cpus] * run_nums,
            "cloud_cpus": [args.cloud_cpus] * run_nums,
            "device_bandwidth": [args.device_bandwidth] * run_nums,
            "edge_bandwidth": [args.edge_bandwidth] * run_nums,
            "cloud_bandwidth": [args.cloud_bandwidth] * run_nums,
            "run_time": deploy_times,
        }
    )

    dataframe = prepare_dataframe(dataframe, add_df)
    dataframe = pd.concat([dataframe, add_df], ignore_index=True)
    dataframe.to_csv(csv_path, index=False)
    
    pass

def usage_test(args : argparse.Namespace, result_folder : str, run_nums : int) :
    csv_path = os.path.join(
        result_folder, "usage.csv"
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
                "edge_cpus",
                "cloud_cpus",
                "device_bandwidth",
                "edge_bandwidth",
                "cloud_bandwidth",
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

    ## Cold Start Runs
    print("🥶 Cold Start Runs")
    for _ in tqdm.tqdm(range(10)):
        inference_caller.call_inference(args.model_name, input_dict)

    ## Hot Runs
    print("🥵 Hot Runs")
    plan_use_times = np.zeros(run_nums)
    for idx in tqdm.tqdm(range(run_nums)):
        _, infer_time, _ = inference_caller.call_inference(args.model_name, input_dict)
        plan_use_times[idx] = infer_time

    add_df = pd.DataFrame(
        {
            "model_name": [args.model_name] * run_nums,
            "latency_weight": [args.latency_weight] * run_nums,
            "energy_weight": [args.energy_weight] * run_nums,
            "device_max_energy": [args.device_max_energy] * run_nums,
            "max_noises": [args.max_noises] * run_nums,
            "device_cpus": [args.device_cpus] * run_nums,
            "edge_cpus": [args.edge_cpus] * run_nums,
            "cloud_cpus": [args.cloud_cpus] * run_nums,
            "device_bandwidth": [args.device_bandwidth] * run_nums,
            "edge_bandwidth": [args.edge_bandwidth] * run_nums,
            "cloud_bandwidth": [args.cloud_bandwidth] * run_nums,
            "run_time": plan_use_times,
        }
    )

    dataframe = prepare_dataframe(dataframe, add_df)
    dataframe = pd.concat([dataframe, add_df], ignore_index=True)
    dataframe.to_csv(csv_path, index=False)

    return plan_use_times


def prepare_dataframe(dataframe: pd.DataFrame, add_df: pd.DataFrame):
    comparison_columns = [
        "model_name",
        "latency_weight",
        "energy_weight",
        "device_max_energy",
        "max_noises",
        "device_cpus",
        "edge_cpus",
        "cloud_cpus",
        "device_bandwidth",
        "edge_bandwidth",
        "cloud_bandwidth",
    ]

    # Filtra righe che NON corrispondono alle tuple in add_df
    dataframe = dataframe[
        ~dataframe[comparison_columns]
        .apply(tuple, axis=1)
        .isin(add_df[comparison_columns].apply(tuple, axis=1))
    ]

    return dataframe


def build_result_folder(has_edge : bool, has_cloud : bool):

    result_folder = ""
    if not has_edge and not has_cloud:
        result_folder = "../Results/DevicePlan"
    elif has_edge and not has_cloud:
        result_folder = "../Results/DeviceEdgePlan"
    elif not has_edge and has_cloud:
        result_folder = "../Results/DeviceCloudPlan"
    else:
        result_folder = "../Results/DeviceEdgeCloudPlan"

    os.makedirs(result_folder, exist_ok=True)


    return result_folder


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


if __name__ == "__main__":
    main()