import argparse
import json

import grpc

from Common import ConfigReader
from CommonPlan.WholePlan import WholePlan
from proto_compiled.common_pb2 import ModelId
from proto_compiled.deployment_pb2 import ProducePlanRequest, ProducePlanResponse
from proto_compiled.deployment_pb2_grpc import DeploymentStub


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
    parser.add_argument(
        "--requests-number", type=int, help="Requests Number", default=1
    )
    parser.add_argument("--max-noises", type=float, help="Max Noises", default=0)
    parser.add_argument("--start-server", type=str, help="Start Server", default="0")

    args = parser.parse_args()
    model_name = args.model
    latency_weight = args.latency_weight
    energy_weight = args.energy_weight
    device_max_energy = args.device_max_energy
    requests_number = args.requests_number
    max_noises = args.max_noises
    start_server = args.start_server

    deployer_addr = ConfigReader.ConfigReader("../../config/config.ini").read_str(
        "addresses", "DEPLOYER_ADDR"
    )
    deployer_port = ConfigReader.ConfigReader("../../config/config.ini").read_int(
        "ports", "DEPLOYER_PORT"
    )

    print(deployer_addr, deployer_port)

    deployer_stub = DeploymentStub(
        grpc.insecure_channel("{}:{}".format(deployer_addr, deployer_port))
    )

    produce_plan_request = ProducePlanRequest(
        models_ids=[ModelId(model_name=model_name)],
        latency_weight=latency_weight,
        energy_weight=energy_weight,
        device_max_energy=device_max_energy,  ## Measured in Joules
        requests_number=[requests_number],
        max_noises=[max_noises],
        start_server=start_server,
    )

    produce_plan_response: ProducePlanResponse = deployer_stub.produce_plan(
        produce_plan_request
    )

    whole_plan = WholePlan.decode(json.loads(produce_plan_response.optimized_plan))

    print(whole_plan.encode())

    # file_name = f"plan_{model_name}_lw_{latency_weight}_ew_{energy_weight}_me_{device_max_energy}_req_{requests_number}_no_{max_noises}.json"

    # with open(file_name, "w") as json_file:
    #     json.dump(whole_plan.encode(), json_file)

    pass


if __name__ == "__main__":
    main()
