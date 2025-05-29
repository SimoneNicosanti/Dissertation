import json

import grpc

from Common import ConfigReader
from CommonPlan.WholePlan import WholePlan
from proto_compiled.common_pb2 import Empty
from proto_compiled.register_pb2_grpc import RegisterStub
from proto_compiled.server_pb2 import AssignmentRequest
from proto_compiled.server_pb2_grpc import AssigneeStub


class FetcherCaller:

    def __init__(self):

        registry_addr = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "addresses", "REGISTRY_ADDR"
        )
        registry_port = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "ports", "REGISTRY_PORT"
        )
        self.registry_chan = grpc.insecure_channel(
            "{}:{}".format(registry_addr, registry_port)
        )
        pass

    def distribute_plan(self, whole_plan: WholePlan):

        assignment_request = AssignmentRequest(
            optimized_plan=json.dumps(whole_plan.encode())
        )

        registry_stub = RegisterStub(self.registry_chan)
        all_server_info = registry_stub.get_all_servers_info(Empty())

        for server_info in all_server_info.all_server_info:
            server_addr = server_info.reachability_info.ip_address
            assignment_port = server_info.reachability_info.assignment_port

            server_chann = grpc.insecure_channel(
                "{}:{}".format(server_addr, assignment_port)
            )

            assignee_stub = AssigneeStub(server_chann)

            assignee_stub.send_plan(assignment_request)

    # def __get_stub_for_assignee(self, assignee_id: str) -> AssigneeStub:

    #     with self.assignee_dict_lock.gen_rlock():
    #         is_present = assignee_id in self.assignee_channels.keys()

    #         if is_present:
    #             channel = self.assignee_channels[assignee_id]

    #     if not is_present:

    #         registry_stub = RegisterStub(self.registry_chan)
    #         reach_info: ReachabilityInfo = registry_stub.get_info_from_id(
    #             ServerId(server_id=assignee_id)
    #         )

    #         with self.assignee_dict_lock.gen_wlock():

    #             if assignee_id in self.assignee_channels.keys():
    #                 channel = self.assignee_channels[assignee_id]
    #             else:
    #                 channel = grpc.insecure_channel(
    #                     f"{reach_info.ip_address}:{reach_info.assignment_port}"
    #                 )
    #                 self.assignee_channels[assignee_id] = channel

    #     return AssigneeStub(channel)
