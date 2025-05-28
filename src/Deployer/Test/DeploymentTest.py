import grpc

from Common import ConfigReader
from proto_compiled.common_pb2 import ModelId
from proto_compiled.deployment_pb2 import DeploymentRequest
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

    deployment_request = DeploymentRequest(
        models_ids=[ModelId(model_name="yolo11n-seg")],
        latency_weight=1,
        energy_weight=1,
        device_max_energy=0,  ## TODO Check this out !!
        requests_number=[1],
        max_noises=[0.1],
        start_server="0",
    )

    deployer_stub.deploy_model(deployment_request)

    pass


if __name__ == "__main__":
    main()
