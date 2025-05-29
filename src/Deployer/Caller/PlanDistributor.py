import grpc
import networkx as nx
from readerwriterlock import rwlock

from Common import ConfigReader
from proto_compiled.common_pb2 import OptimizedPlan
from proto_compiled.register_pb2 import ReachabilityInfo, ServerId
from proto_compiled.register_pb2_grpc import RegisterStub
from proto_compiled.server_pb2_grpc import AssigneeStub


class PlanDistributor:

    def __init__(self):
        self.assignee_dict_lock = rwlock.RWLockWriteD()
        self.assignee_channels: dict[str, grpc.Channel] = {}

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

    def distribute_plan(
        self,
        plan_dict: dict[str, str],
        network_graph: nx.DiGraph,
        deployment_server: str,
    ):

        optimized_plan = OptimizedPlan(
            deployer_id=deployment_server, plans_map=plan_dict
        )
        for net_node_id in network_graph.nodes:

            assignee_stub = self.__get_stub_for_assignee(net_node_id.node_name)
            assignee_stub.send_plan(optimized_plan)

    def __get_stub_for_assignee(self, assignee_id: str) -> AssigneeStub:

        with self.assignee_dict_lock.gen_rlock():
            is_present = assignee_id in self.assignee_channels.keys()

            if is_present:
                channel = self.assignee_channels[assignee_id]

        if not is_present:

            registry_stub = RegisterStub(self.registry_chan)
            reach_info: ReachabilityInfo = registry_stub.get_info_from_id(
                ServerId(server_id=assignee_id)
            )

            with self.assignee_dict_lock.gen_wlock():

                if assignee_id in self.assignee_channels.keys():
                    channel = self.assignee_channels[assignee_id]
                else:
                    channel = grpc.insecure_channel(
                        f"{reach_info.ip_address}:{reach_info.assignment_port}"
                    )
                    self.assignee_channels[assignee_id] = channel

        return AssigneeStub(channel)
