import grpc
from readerwriterlock import rwlock

from Optimizer.Graph.NetworkGraph import NetworkGraph
from proto_compiled.common_pb2 import OptimizedPlan
from proto_compiled.register_pb2 import ReachabilityInfo, ServerId
from proto_compiled.register_pb2_grpc import RegisterStub
from proto_compiled.server_pb2_grpc import AssigneeStub

REGISTRY_PORT = 50051


class PlanDistributor:

    def __init__(self):
        self.assignee_dict_lock = rwlock.RWLockWriteD()
        self.assignee_channels: dict[str, grpc.Channel] = {}

        self.registry_chan = grpc.insecure_channel("registry:{}".format(REGISTRY_PORT))
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
