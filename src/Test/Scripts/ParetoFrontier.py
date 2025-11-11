import argparse
import json
import os

import grpc
import numpy as np
import pandas as pd
import tqdm

from Common.ConfigReader import ConfigReader
from CommonPlan.Plan import Plan
from CommonPlan.WholePlan import WholePlan
from proto_compiled.common_pb2 import ModelId
from proto_compiled.deployment_pb2 import ProducePlanRequest, ProducePlanResponse
from proto_compiled.deployment_pb2_grpc import DeploymentStub

STEP = 0.01


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


def generate_plan(
    deployer_stub,
    model_name,
    latency_weight,
    energy_weight,
    device_max_energy=0.0,
    max_noises=0.0,
) -> Plan:
    produce_plan_request = ProducePlanRequest(
        models_ids=[ModelId(model_name=model_name)],
        latency_weight=latency_weight,
        energy_weight=energy_weight,
        device_max_energy=device_max_energy,
        requests_number=[1],
        max_noises=[max_noises],
        start_server="0",
    )

    produce_plan_res: ProducePlanResponse = deployer_stub.produce_plan(
        produce_plan_request
    )

    whole_plan: WholePlan = WholePlan.decode(
        json.loads(produce_plan_res.optimized_plan)
    )
    plan: Plan = whole_plan.get_model_plan(model_name)

    return plan


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-name", type=str, help="Model Name", required=True)
    argparser.add_argument(
        "--device-cpus", type=float, help="Device CPUs", required=True
    )
    argparser.add_argument("--edge-cpus", type=float, help="Edge CPUs", required=True)
    argparser.add_argument(
        "--device-bandwidth", type=float, help="Device Bandwidth", required=True
    )
    argparser.add_argument(
        "--edge-bandwidth", type=float, help="Edge Bandwidth", required=True
    )
    args = argparser.parse_args()

    os.makedirs("../Results/ParetoFrontier", exist_ok=True)

    latency_weights = np.arange(0, 1 + STEP, STEP)

    deployer_stub = get_deployer_stub()

    if os.path.exists("../Results/ParetoFrontier/pareto_frontier.csv"):
        df = pd.read_csv("../Results/ParetoFrontier/pareto_frontier.csv")
    else:
        df = pd.DataFrame(
            columns=[
                "model_name",
                "device_cpus",
                "edge_cpus",
                "device_bandwidth",
                "edge_bandwidth",
                "latency_weight",
                "latency_cost",
                "energy_cost",
            ]
        )

    common_cols = [
        "model_name",
        "device_cpus",
        "edge_cpus",
        "device_bandwidth",
        "edge_bandwidth",
    ]
    values = [
        args.model_name,
        args.device_cpus,
        args.edge_cpus,
        args.device_bandwidth,
        args.edge_bandwidth,
    ]
    df = df[~df[common_cols].isin(values).all(axis=1)]

    for lw in tqdm.tqdm(latency_weights):  # latency_weights:
        curr_plan: Plan = generate_plan(
            deployer_stub, args.model_name, lw, 1 - lw, max_noises=0.5
        )

        latency_cost = curr_plan.get_latency_cost()
        energy_cost = curr_plan.get_energy_cost()

        new_row = [
            args.model_name,
            args.device_cpus,
            args.edge_cpus,
            args.device_bandwidth,
            args.edge_bandwidth,
            lw,
            latency_cost,
            energy_cost,
        ]
        df.loc[len(df)] = new_row

    df.to_csv("../Results/ParetoFrontier/pareto_frontier.csv", index=False)

    pass


if __name__ == "__main__":
    main()
