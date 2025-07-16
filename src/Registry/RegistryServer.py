import json
import time

import networkx as nx
from readerwriterlock import rwlock

from CommonIds.NodeId import NodeId
from CommonProfile.NetworkInfo import NetworkEdgeInfo, NetworkNodeInfo
from CommonProfile.NetworkProfile import NetworkProfile
from proto_compiled.common_pb2 import Empty
from proto_compiled.register_pb2 import (
    AllServerInfo,
    ReachabilityInfo,
    ServerId,
    ServerInfo,
    ServerState,
    StateMap,
)
from proto_compiled.register_pb2_grpc import RegisterServicer


class RegistryServer(RegisterServicer):

    def __init__(
        self,
    ):
        self.reach_dict_lock = rwlock.RWLockWriteD()
        self.reachability_dict: dict[str, tuple[ReachabilityInfo, int]] = {}

        self.server_id_lock = rwlock.RWLockWriteD()
        self.server_id = 0

        self.network_graph_lock = rwlock.RWLockWriteD()
        self.network_graph: nx.DiGraph = nx.DiGraph()
        self.state_push_times = {}  ## TODO Check keep alive

        pass

    def register_server(self, reachability_info: ReachabilityInfo, context) -> ServerId:
        with self.reach_dict_lock.gen_rlock():
            if reachability_info.ip_address in self.reachability_dict:
                print("Server already registered")
                server_id = self.reachability_dict[reachability_info.ip_address][1]
                return ServerId(server_id=str(server_id))

        with self.server_id_lock.gen_wlock():
            new_server_id = str(self.server_id)
            self.server_id += 1

        with self.reach_dict_lock.gen_wlock():
            self.reachability_dict[reachability_info.ip_address] = [
                reachability_info,
                new_server_id,
            ]

        register_response = ServerId(server_id=new_server_id)
        self.log_server_registration(reachability_info, new_server_id)

        return register_response

    def get_all_servers_info(self, request: Empty, context) -> AllServerInfo:

        server_info_list: list[ServerInfo] = []
        with self.reach_dict_lock.gen_rlock():
            for reach_info, server_id in self.reachability_dict.values():
                server_info = ServerInfo(
                    reachability_info=reach_info,
                    server_id=ServerId(server_id=server_id),
                )
                server_info_list.append(server_info)

        return AllServerInfo(all_server_info=server_info_list)

    def get_info_from_id(self, server_id: ServerId, context) -> ReachabilityInfo:
        with self.reach_dict_lock.gen_rlock():
            for reach_info, id in self.reachability_dict.values():
                if str(id) == server_id.server_id:
                    return reach_info

        raise Exception("Server not found")

    def log_server_registration(
        self, reachability_info: ReachabilityInfo, server_id: int
    ) -> None:
        print("Received Registration")
        print(f"\t IP Addr >> {reachability_info.ip_address}")
        print(f"\t Assignee Port >> {reachability_info.assignment_port}")
        print(f"\t Inference Port >> {reachability_info.inference_port}")
        print(f"\t Ping Port >> {reachability_info.ping_port}")
        print(f"\t Server ID >> {server_id}")
        print("----------------------------------------")

    def pull_all_states(self, empty_req, context) -> StateMap:

        with self.network_graph_lock.gen_rlock():
            network_profile = NetworkProfile()
            network_profile.set_network_graph(self.network_graph)
            network_profile_dict = network_profile.encode()
            network_profile_dict_str = json.dumps(network_profile_dict)
            state_map = StateMap(network_profile=network_profile_dict_str)

            return state_map

    def push_state(self, server_state: ServerState, context):
        with self.network_graph_lock.gen_wlock():
            self.state_push_times[server_state.server_id] = time.time()
            ## Update Network State
            self.update_network_graph(server_state.server_id, server_state.state)

            print("=== Network Graph Summary ===")
            print("Nodes:")
            for node, data in self.network_graph.nodes(data=True):
                print(
                    f"  Node {node}: mem={data['available_memory']:.1f} MB, "
                    f"comp={data['comp_energy_per_sec']:.3f} W, "
                    f"trans_ps={data['trans_energy_per_sec']:.3f} W, "
                    f"trans_base={data['trans_energy_base']:.3f} W"
                )

            print("\nEdges:")
            for u, v, data in self.network_graph.edges(data=True):
                print(
                    f"  Edge ({u} → {v}): lat={data['latency']:.4f} s, bw={data['bandwidth']:.2f} MB/s"
                )
            print("=== End Graph Summary ===\n")

        return Empty()

    def update_network_graph(self, server_id: str, server_state: str):
        state_dict = json.loads(server_state)

        node_id = NodeId(node_name=server_id)
        self.network_graph.add_node(node_id)

        self.network_graph.nodes[node_id][NetworkNodeInfo.AVAILABLE_MEMORY] = (
            state_dict["available_memory"]
        )

        self.network_graph.nodes[node_id][NetworkNodeInfo.COMP_ENERGY_PER_SEC] = (
            state_dict["comp_energy_per_sec"]
        )

        self.network_graph.nodes[node_id][NetworkNodeInfo.TRANS_ENERGY_PER_SEC] = (
            state_dict["trans_energy_per_sec"]
        )
        self.network_graph.nodes[node_id][NetworkNodeInfo.TRANS_ENERGY_BASE] = (
            state_dict["trans_energy_base"]
        )

        self.network_graph.nodes[node_id][NetworkNodeInfo.SELF_TRANS_ENERGY_PER_SEC] = (
            state_dict["self_trans_energy_per_sec"]
        )
        self.network_graph.nodes[node_id][NetworkNodeInfo.SELF_TRANS_ENERGY_BASE] = (
            state_dict["self_trans_energy_base"]
        )

        self.network_graph.nodes[node_id][NetworkNodeInfo.IDX] = int(server_id)

        latencies: dict = state_dict["latencies"]
        bandwidths = state_dict["bandwidths"]

        to_rmv_edges = set()
        for edge in self.network_graph.edges:
            if edge[0] == node_id:
                to_rmv_edges.add(edge)

        for edge in to_rmv_edges:
            self.network_graph.remove_edge(edge[0], edge[1])

        for next_server_id in latencies.keys():
            next_node_id = NodeId(node_name=next_server_id)

            edge_latency = latencies[next_server_id]
            edge_bandwidth = bandwidths[next_server_id]

            self.network_graph.add_edge(node_id, next_node_id)

            self.network_graph.edges[node_id, next_node_id][
                NetworkEdgeInfo.LATENCY
            ] = edge_latency

            self.network_graph.edges[node_id, next_node_id][
                NetworkEdgeInfo.BANDWIDTH
            ] = edge_bandwidth
