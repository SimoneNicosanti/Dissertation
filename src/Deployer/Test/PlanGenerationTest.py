import json

import grpc

from Common import ConfigReader
from CommonPlan.WholePlan import WholePlan
from proto_compiled.common_pb2 import ModelId
from proto_compiled.deployment_pb2 import ProducePlanRequest, ProducePlanResponse
from proto_compiled.deployment_pb2_grpc import DeploymentStub


def main():
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
        models_ids=[ModelId(model_name="yolo11x-seg")],
        latency_weight=1,
        energy_weight=0.0,
        device_max_energy=0,  ## Measured in Joules
        requests_number=[1],
        max_noises=[2],
        start_server="0",
    )

    produce_plan_response: ProducePlanResponse = deployer_stub.produce_plan(
        produce_plan_request
    )

    whole_plan = WholePlan.decode(json.loads(produce_plan_response.optimized_plan))

    print(whole_plan.encode())

    with open("whole_plan.json", "w") as json_file:
        json.dump(whole_plan.encode(), json_file)

    pass


if __name__ == "__main__":
    main()
