import grpc

from Optimizer.Graph.NetworkGraph import NetworkEdgeInfo, NetworkGraph, NetworkNodeInfo
from proto_compiled.common_pb2 import Empty
from proto_compiled.register_pb2 import AllServerInfo
from proto_compiled.register_pb2_grpc import RegisterStub

## TODO Implement Server Monitoring!!
## Do not contact the registry but the global monitor receiving the state from all servers


class NetworkBuilder:
    def __init__(self):

        self.registry_connection = grpc.insecure_channel("registry:50051")
        pass

    def build_network(self) -> NetworkGraph:
        registry_stub = RegisterStub(self.registry_connection)
        all_server_info: AllServerInfo = registry_stub.get_all_servers_info(Empty())

        graph = NetworkGraph("NetworkGraph")
        for server_info in all_server_info.all_server_info:
            node_id = graph.build_node_id(server_info.server_id.server_id)
            if server_info.server_id.server_id == "0":
                node_info = NetworkNodeInfo(
                    net_node_flops_per_sec=4 * 10**9,
                    net_node_comp_energy_per_sec=0.5,
                    net_node_trans_energy_per_sec=0.5,
                    net_node_available_memory=10,
                    net_node_ip_address=server_info.reachability_info.ip_address,
                    net_node_port=server_info.reachability_info.assignment_port,
                )
            elif server_info.server_id.server_id == "1":
                node_info = NetworkNodeInfo(
                    net_node_flops_per_sec=5 * 10**9,
                    net_node_comp_energy_per_sec=1,
                    net_node_trans_energy_per_sec=1,
                    net_node_available_memory=100_000_000,
                    net_node_ip_address=server_info.reachability_info.ip_address,
                    net_node_port=server_info.reachability_info.assignment_port,
                )
            else:
                continue
            graph.put_node(node_id, node_info)

        for server_info_1 in all_server_info.all_server_info:
            for server_info_2 in all_server_info.all_server_info:

                if (
                    server_info_1.server_id.server_id
                    == server_info_2.server_id.server_id
                ):
                    bandwidth = None

                else:
                    if (
                        server_info_1.server_id.server_id == "0"
                        and server_info_2.server_id.server_id == "1"
                    ):
                        bandwidth = 1.5
                    elif (
                        server_info_1.server_id.server_id == "1"
                        and server_info_2.server_id.server_id == "0"
                    ):
                        bandwidth = 1.5
                    else:
                        continue

                edge_info = NetworkEdgeInfo(net_edge_bandwidth=bandwidth)
                edge_id = graph.build_edge_id(
                    server_info_1.server_id.server_id, server_info_2.server_id.server_id
                )

                graph.put_edge(edge_id, edge_info)

        return graph
