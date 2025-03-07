import grpc
from Graph.NetworkGraph import NetworkGraph, NetworkNodeInfo
from Plan.Plan import Plan
from proto.server_pb2_grpc import AssigneeStub
from proto.common_pb2 import OptimizedPlan


class PlanDistributor:

    def __init__(self):
        self.stubs_dict: dict[str, AssigneeStub] = {}
        pass

    def distribute_plan(self, plan_dict: dict[str, str], network_graph: NetworkGraph):

        optimized_plan = OptimizedPlan(plans_map=plan_dict)
        for net_node_id in network_graph.get_nodes_id():
            network_node_info: NetworkNodeInfo = network_graph.get_node_info(
                net_node_id
            )

            assignee_ip_addr = network_node_info.get_assignee_ip_addr()
            assignee_port = network_node_info.get_assignee_port()

            if assignee_ip_addr is None or assignee_port is None:
                continue

            assignee_stub = self.stubs_dict.get(assignee_ip_addr)
            if assignee_stub is None:
                self.stubs_dict[assignee_ip_addr] = AssigneeStub(
                    grpc.insecure_channel(f"{assignee_ip_addr}:{assignee_port}")
                )
                assignee_stub = self.stubs_dict[assignee_ip_addr]

            assignee_stub.send_plan(optimized_plan)

        pass
