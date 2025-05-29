import itertools

import networkx as nx

from CommonIds.NodeId import NodeId
from CommonProfile.NetworkInfo import NetworkNodeInfo
from CommonProfile.NetworkProfile import NetworkProfile


class NetworkProfileBuilder:
    def __init__(self):

        ## Connect to InfluxDB
        ## Retrieve Network Info
        ## Connect to find static server info (energy consumptions)

        pass

    def build_network_profile(self) -> NetworkProfile:

        network_graph: nx.DiGraph = nx.DiGraph()
        for i, j in itertools.product(range(2), repeat=2):
            first_node = NodeId(node_name=str(i))
            second_node = NodeId(node_name=str(j))
            if i == j:
                network_graph.add_edge(first_node, second_node, bandwidth=100)
                network_graph.add_edge(first_node, second_node, latency=0)

            else:
                network_graph.add_edge(first_node, second_node, bandwidth=7.5)
                network_graph.add_edge(first_node, second_node, latency=0.5)

                network_graph.add_edge(second_node, first_node, bandwidth=7.5)
                network_graph.add_edge(second_node, first_node, latency=0.5)

        for i in range(2):
            node_id = NodeId(node_name=str(i))
            if i == 0:
                network_graph.nodes[node_id][NetworkNodeInfo.FLOPS_PER_SEC] = 2 * 10**9
                network_graph.nodes[node_id][NetworkNodeInfo.COMP_ENERGY_PER_SEC] = 2
                network_graph.nodes[node_id][NetworkNodeInfo.TRANS_ENERGY_PER_SEC] = 1
                network_graph.nodes[node_id][NetworkNodeInfo.AVAILABLE_MEMORY] = 1_000
                network_graph.nodes[node_id][NetworkNodeInfo.IDX] = i
            else:
                network_graph.nodes[node_id][NetworkNodeInfo.FLOPS_PER_SEC] = 5 * 10**9
                network_graph.nodes[node_id][NetworkNodeInfo.COMP_ENERGY_PER_SEC] = 10
                network_graph.nodes[node_id][NetworkNodeInfo.TRANS_ENERGY_PER_SEC] = 2
                network_graph.nodes[node_id][NetworkNodeInfo.AVAILABLE_MEMORY] = 10_000
                network_graph.nodes[node_id][NetworkNodeInfo.IDX] = i

        network_profile = NetworkProfile()
        network_profile.set_network_graph(network_graph)

        return network_profile
        pass

    # def do_bandwidths_retrieval(self):
    #     ## Call InfluxDB stats get
    #     ## Or call stats collector get

    #     registry_stub = RegisterStub(self.registry_chann)
    #     all_server_info: AllServerInfo = registry_stub.get_all_servers_info(Empty())

    #     bandwidths = {}
    #     latencies = {}

    #     for server_info_src in all_server_info.all_server_info:
    #         for server_info_dst in all_server_info.all_server_info:
    #             if (
    #                 server_info_src.server_id.server_id
    #                 != server_info_dst.server_id.server_id
    #             ):
    #                 bandwidths[
    #                     (
    #                         server_info_src.server_id.SerializeToString(),
    #                         server_info_dst.server_id.SerializeToString(),
    #                     )
    #                 ] = 1_000_000
    #                 latencies[
    #                     (
    #                         server_info_src.server_id.SerializeToString(),
    #                         server_info_dst.server_id.SerializeToString(),
    #                     )
    #                 ] = 0
    #             else:
    #                 bandwidths[
    #                     (
    #                         server_info_src.server_id.SerializeToString(),
    #                         server_info_dst.server_id.SerializeToString(),
    #                     )
    #                 ] = 10
    #                 latencies[
    #                     (
    #                         server_info_src.server_id.SerializeToString(),
    #                         server_info_dst.server_id.SerializeToString(),
    #                     )
    #                 ] = 0.5

    #     pass

    # def build_network_graph(self, bandwidths, latencies, server_profiles):
    #     network_graph = nx.DiGraph()
    #     for server_id_src, server_id_dst in bandwidths.keys():
    #         network_graph.add_edge(server_id_src, server_id_dst)
    #         network_graph[server_id_src][server_id_dst]["bandwidth"] = bandwidths[
    #             (server_id_src, server_id_dst)
    #         ]
    #         network_graph[server_id_src][server_id_dst]["latency"] = latencies[
    #             (server_id_src, server_id_dst)
    #         ]

    #     for server_id in network_graph.nodes:
    #         network_graph.nodes[server_id]["flops_per_sec"] = server_profiles[
    #             server_id
    #         ]["flops_per_sec"]
    #         network_graph.nodes[server_id]["comp_energy_per_sec"] = server_profiles[
    #             server_id
    #         ]["comp_energy_per_sec"]
    #         network_graph.nodes[server_id]["trans_energy_per_sec"] = server_profiles[
    #             server_id
    #         ]["trans_energy_per_sec"]
    #         network_graph.nodes[server_id]["available_memory"] = server_profiles[
    #             server_id
    #         ]["available_memory"]

    #     return network_graph
