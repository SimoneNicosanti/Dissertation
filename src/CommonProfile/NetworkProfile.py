import networkx as nx

from CommonIds.NodeId import NodeId


class NetworkProfile:

    def __init__(self) -> None:
        self.network_graph: nx.DiGraph = None
        pass

    def set_network_graph(self, network_graph: nx.DiGraph) -> None:
        self.network_graph = network_graph

    def get_network_graph(self) -> nx.DiGraph:
        return self.network_graph

    def encode(self) -> dict:
        profile_dict = {}

        if self.network_graph is not None:
            label_mapping = {
                node_id: node_id.node_name for node_id in self.network_graph.nodes
            }
            relabeled_graph = nx.relabel_nodes(
                self.network_graph, label_mapping, copy=True
            )

            encoded_graph = nx.json_graph.node_link_data(relabeled_graph)

            profile_dict["graph"] = encoded_graph

        return profile_dict

    @staticmethod
    def decode(network_profile_dict: dict) -> "NetworkProfile":

        decoded_profile = NetworkProfile()

        network_graph = nx.json_graph.node_link_graph(network_profile_dict["graph"])
        network_graph = nx.relabel_nodes(
            network_graph,
            {node_name: NodeId(node_name) for node_name in network_graph.nodes},
            copy=True,
        )
        decoded_profile.set_network_graph(network_graph)

        return decoded_profile
