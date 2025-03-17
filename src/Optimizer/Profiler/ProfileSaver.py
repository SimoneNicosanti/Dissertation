import copy
import json
import os

import networkx as nx

from Optimizer.Graph.Graph import NodeId

MODEL_PROFILES_DIR = "/optimizer_data/models_profiles/"


def save_profile(
    model_graph: nx.DiGraph,
):
    to_save_graph = copy.deepcopy(model_graph)
    label_mapping = {node_id: node_id.node_name for node_id in to_save_graph.nodes}
    to_save_graph = nx.relabel_nodes(to_save_graph, label_mapping)
    model_name = model_graph.graph["name"]
    model_profile_path = os.path.join(MODEL_PROFILES_DIR, model_name + ".json")
    with open(model_profile_path, "w") as json_file:
        json.dump(nx.node_link_data(to_save_graph, edges="links"), json_file)
    pass


def read_profile(model_name: str) -> nx.DiGraph:
    model_profile_path = os.path.join(MODEL_PROFILES_DIR, model_name + ".json")
    if os.path.isfile(model_profile_path):
        with open(model_profile_path, "r") as json_file:
            model_graph: nx.MultiDiGraph = nx.node_link_graph(
                json.load(json_file), edges="links"
            )

        label_mapping = {
            node_name: NodeId(node_name) for node_name in model_graph.nodes
        }
        model_graph = nx.relabel_nodes(model_graph, label_mapping)
        return model_graph
    return None
