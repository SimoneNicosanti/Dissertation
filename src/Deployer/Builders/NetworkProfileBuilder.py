import json

import grpc
import networkx as nx

from Common.ConfigReader import ConfigReader
from CommonIds.NodeId import NodeId
from CommonProfile.NetworkInfo import NetworkEdgeInfo, NetworkNodeInfo
from CommonProfile.NetworkProfile import NetworkProfile
from proto_compiled.common_pb2 import Empty
from proto_compiled.state_pool_pb2 import StateMap
from proto_compiled.state_pool_pb2_grpc import StatePoolStub


class NetworkProfileBuilder:
    def __init__(self):

        ## Connect to InfluxDB
        ## Retrieve Network Info
        ## Connect to find static server info (energy consumptions)
        state_pool_addr = ConfigReader().read_str("addresses", "STATE_POOL_ADDR")
        state_pool_port = ConfigReader().read_int("ports", "STATE_POOL_PORT")
        self.state_pool_conn = grpc.insecure_channel(
            "{}:{}".format(state_pool_addr, state_pool_port)
        )

        pass

    def build_network_profile(self) -> NetworkProfile:
        network_graph: nx.DiGraph = nx.DiGraph()

        state_pool_stub = StatePoolStub(self.state_pool_conn)
        pull_response: StateMap = state_pool_stub.pull_all_states(Empty())
        state_map = pull_response.state_map

        for server_id, server_state in state_map.items():
            server_state = json.loads(server_state)
            node_id = NodeId(node_name=server_id)
            network_graph.add_node(node_id)

            network_graph.nodes[node_id][NetworkNodeInfo.AVAILABLE_MEMORY] = 100

            network_graph.nodes[node_id][NetworkNodeInfo.FLOPS_PER_SEC] = 0

            if server_id == "0":
                comp_energy = 10  ## W = J / s

                trans_energy = 6.0
                trans_energy_base = 1.2

                self_trans_energy = 0.0
                self_trans_energy_base = 0.0
            else:
                comp_energy = 15.0

                trans_energy = 15.6
                trans_energy_base = 1.2

                self_trans_energy = 0.0
                self_trans_energy_base = 0.0

            network_graph.nodes[node_id][
                NetworkNodeInfo.COMP_ENERGY_PER_SEC
            ] = comp_energy

            network_graph.nodes[node_id][
                NetworkNodeInfo.TRANS_ENERGY_PER_SEC
            ] = trans_energy

            network_graph.nodes[node_id][
                NetworkNodeInfo.TRANS_ENERGY_BASE
            ] = trans_energy_base

            network_graph.nodes[node_id][
                NetworkNodeInfo.SELF_TRANS_ENERGY_PER_SEC
            ] = self_trans_energy

            network_graph.nodes[node_id][
                NetworkNodeInfo.SELF_TRANS_ENERGY_BASE
            ] = self_trans_energy_base

            network_graph.nodes[node_id][NetworkNodeInfo.IDX] = int(server_id)

        for src_server_id, src_server_state in state_map.items():
            src_node_id = NodeId(node_name=src_server_id)
            src_server_state = json.loads(src_server_state)
            print(src_server_state["bandwidths"])
            for dst_server_id in state_map.keys():
                dst_node_id = NodeId(node_name=dst_server_id)

                network_graph.add_edge(src_node_id, dst_node_id)

                network_graph.edges[src_node_id, dst_node_id][
                    NetworkEdgeInfo.BANDWIDTH
                ] = src_server_state["bandwidths"][dst_server_id]

                network_graph.edges[src_node_id, dst_node_id][
                    NetworkEdgeInfo.LATENCY
                ] = src_server_state["latencies"][dst_server_id]

        network_profile = NetworkProfile()
        network_profile.set_network_graph(network_graph)

        return network_profile

    # def build_network_profile(self) -> NetworkProfile:

    #     network_graph: nx.DiGraph = nx.DiGraph()
    #     for i, j in itertools.product(range(2), repeat=2):
    #         first_node = NodeId(node_name=str(i))
    #         second_node = NodeId(node_name=str(j))
    #         if i == j:
    #             network_graph.add_edge(first_node, second_node, bandwidth=50)
    #             network_graph.add_edge(first_node, second_node, latency=0)

    #         else:
    #             network_graph.add_edge(first_node, second_node, bandwidth=7.5)
    #             network_graph.add_edge(first_node, second_node, latency=0.05)

    #             network_graph.add_edge(second_node, first_node, bandwidth=7.5)
    #             network_graph.add_edge(second_node, first_node, latency=0.05)

    #     for i in range(2):
    #         node_id = NodeId(node_name=str(i))
    #         if i == 0:
    #             network_graph.nodes[node_id][NetworkNodeInfo.FLOPS_PER_SEC] = 2 * 10**9
    #             network_graph.nodes[node_id][NetworkNodeInfo.COMP_ENERGY_PER_SEC] = 2
    #             network_graph.nodes[node_id][NetworkNodeInfo.TRANS_ENERGY_PER_SEC] = 1
    #             network_graph.nodes[node_id][NetworkNodeInfo.AVAILABLE_MEMORY] = 1_000
    #             network_graph.nodes[node_id][NetworkNodeInfo.IDX] = i
    #         else:
    #             network_graph.nodes[node_id][NetworkNodeInfo.FLOPS_PER_SEC] = 5 * 10**9
    #             network_graph.nodes[node_id][NetworkNodeInfo.COMP_ENERGY_PER_SEC] = 10
    #             network_graph.nodes[node_id][NetworkNodeInfo.TRANS_ENERGY_PER_SEC] = 2
    #             network_graph.nodes[node_id][NetworkNodeInfo.AVAILABLE_MEMORY] = 10_000
    #             network_graph.nodes[node_id][NetworkNodeInfo.IDX] = i

    #     network_profile = NetworkProfile()
    #     network_profile.set_network_graph(network_graph)

    #     return network_profile
    #     pass

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
