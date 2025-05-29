import json

import grpc

from Common.ConfigReader import ConfigReader
from CommonPlan.WholePlan import WholePlan
from proto_compiled.deployment_pb2 import DeploymentRequest
from proto_compiled.deployment_pb2_grpc import DeploymentStub


def main() -> None:

    with open("whole_plan.json", "r") as f:
        whole_plane: WholePlan = WholePlan.decode(json.load(f))

    deployer_addr = ConfigReader("../../config/config.ini").read_str(
        "addresses", "DEPLOYER_ADDR"
    )
    deployer_port = ConfigReader("../../config/config.ini").read_int(
        "ports", "DEPLOYER_PORT"
    )

    print(deployer_addr, deployer_port)

    deployer_stub = DeploymentStub(
        grpc.insecure_channel("{}:{}".format(deployer_addr, deployer_port))
    )

    deployment_request = DeploymentRequest(
        optimized_plan=json.dumps(whole_plane.encode())
    )

    deployer_stub.deploy_plan(deployment_request)


if __name__ == "__main__":
    main()
    pass
