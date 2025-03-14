import grpc
from readerwriterlock import rwlock

from Optimizer.Graph.NetworkGraph import NetworkGraph, NetworkNodeInfo
from proto_compiled.common_pb2 import OptimizedPlan
from proto_compiled.server_pb2_grpc import AssigneeStub


class PlanDistributor:

    def __init__(self):
        self.assignee_dict_lock = rwlock.RWLockWriteD()
        self.assignee_channels: dict[str, grpc.Channel] = {}
        pass

    def distribute_plan(
        self,
        plan_dict: dict[str, str],
        network_graph: NetworkGraph,
        deployment_server: str,
    ):

        optimized_plan = OptimizedPlan(
            deployer_id=deployment_server, plans_map=plan_dict
        )
        for net_node_id in network_graph.get_nodes_id():
            network_node_info: NetworkNodeInfo = network_graph.get_node_info(
                net_node_id
            )

            assignee_ip_addr = network_node_info.get_assignee_ip_addr()
            assignee_port = network_node_info.get_assignee_port()

            if assignee_ip_addr is None or assignee_port is None:
                continue

            assignee_stub = self.__get_stub_for_assignee(
                assignee_ip_addr, assignee_port
            )

            assignee_stub.send_plan(optimized_plan)

    def __get_stub_for_assignee(
        self, assignee_ip_addr: str, assignee_port: int
    ) -> AssigneeStub:

        with self.assignee_dict_lock.gen_rlock():
            is_present = assignee_ip_addr in self.assignee_channels.keys()

            if is_present:
                channel = self.assignee_channels[assignee_ip_addr]

        if not is_present:
            with self.assignee_dict_lock.gen_wlock():

                if assignee_ip_addr in self.assignee_channels.keys():
                    channel = self.assignee_channels[assignee_ip_addr]
                else:
                    channel = grpc.insecure_channel(
                        f"{assignee_ip_addr}:{assignee_port}"
                    )
                    self.assignee_channels[assignee_ip_addr] = channel

        return AssigneeStub(channel)
