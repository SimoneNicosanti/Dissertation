import argparse
import time

import grpc
import numpy as np

from Common.ConfigReader import ConfigReader
from proto_compiled.common_pb2 import ModelId
from proto_compiled.deployment_pb2 import (
    DeploymentRequest,
    ProducePlanRequest,
    ProducePlanResponse,
)
from proto_compiled.deployment_pb2_grpc import DeploymentStub


def get_deployer_stub():
    deployer_addr = ConfigReader.ConfigReader("../../config/config.ini").read_str(
        "addresses", "DEPLOYER_ADDR"
    )
    deployer_port = ConfigReader.ConfigReader("../../config/config.ini").read_int(
        "ports", "DEPLOYER_PORT"
    )
    deployer_stub = DeploymentStub(
        grpc.insecure_channel("{}:{}".format(deployer_addr, deployer_port))
    )
    return deployer_stub


def main():

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
    parser.add_argument("--max-noises", type=int, help="Max Noises", default=0)

    parser.add_argument("--plan-gen-runs", type=int, help="Plan Gen Runs", default=1)
    parser.add_argument(
        "--plan-deploy-runs", type=int, help="Plan Deploy Runs", default=1
    )
    parser.add_argument("--plan-use-runs", type=int, help="Plan Use Runs", default=60)

    args = parser.parse_args()
    model_name = args.model
    latency_weight = args.latency_weight
    energy_weight = args.energy_weight
    device_max_energy = args.device_max_energy
    max_noises = args.max_noises

    plan_gen_runs = args.plan_gen_runs
    plan_deploy_runs = args.plan_deploy_runs
    plan_use_runs = args.plan_use_runs

    ## Plan Production

    produce_plan_request = ProducePlanRequest(
        model_ids=[ModelId(model_name=model_name)],
        latency_weight=latency_weight,
        energy_weight=energy_weight,
        device_max_energy=device_max_energy,
        max_noises=max_noises,
    )

    deployer_stub = get_deployer_stub()

    produce_plan_times = np.zeros(plan_gen_runs)
    for idx in range(plan_gen_runs):
        start = time.perf_counter_ns()
        produce_plan_response: ProducePlanResponse = deployer_stub.produce_plan(
            produce_plan_request
        )
        end = time.perf_counter_ns()
        produce_plan_times[idx] = (end - start) * 1e-9

    ## Plan Deployment

    deployment_request = DeploymentRequest(
        optimized_plan=produce_plan_response.optimized_plan
    )

    deployment_plan_times = np.zeros(plan_deploy_runs)
    for idx in range(plan_deploy_runs):
        start = time.perf_counter_ns()
        deployer_stub.deploy_plan(deployment_request)
        end = time.perf_counter_ns()
        deployment_plan_times[idx] = (end - start) * 1e-9

    pass


if __name__ == "__main__":
    main()
