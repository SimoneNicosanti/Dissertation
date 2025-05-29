import json

import grpc

from Common import ConfigReader
from CommonPlan.WholePlan import WholePlan
from proto_compiled.model_divide_pb2 import PartitionRequest
from proto_compiled.model_divide_pb2_grpc import ModelDivideStub


class ModelDivider:

    def __init__(self) -> None:
        divider_addr = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "addresses", "MODEL_DIVIDER_ADDR"
        )
        divider_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
            "ports", "MODEL_DIVIDER_PORT"
        )
        self.divider_chann = grpc.insecure_channel(
            "{}:{}".format(divider_addr, divider_port)
        )

    def divide_model(self, whole_plan: WholePlan):

        model_divider_stub = ModelDivideStub(self.divider_chann)

        partition_request = PartitionRequest(
            optimized_plan=json.dumps(whole_plan.encode())
        )

        model_divider_stub.divide_model(partition_request)

        pass
