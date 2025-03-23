import copy
import json
import os
import profile

import networkx as nx

from Common import ConfigReader
from CommonProfile.NodeId import NodeId



def save_profile(
    model_graph: nx.DiGraph,
):
    profiles_dir = ConfigReader.ConfigReader("./config/config.ini").read_str(
        "model_manager_dirs", "PROFILES_DIR"
    )
    to_save_graph = copy.deepcopy(model_graph)
    label_mapping = {node_id: node_id.node_name for node_id in to_save_graph.nodes}
    to_save_graph = nx.relabel_nodes(to_save_graph, label_mapping)
    model_name = model_graph.graph["name"]
    model_profile_path = os.path.join(profiles_dir, model_name + ".json")
    
    with open(model_profile_path, "w") as json_file:
        json.dump(nx.node_link_data(to_save_graph), json_file)
    pass


def read_profile(model_name: str) -> nx.DiGraph:
    profiles_dir = ConfigReader.ConfigReader("./config/config.ini").read_str(
        "model_manager_dirs", "PROFILES_DIR"
    )

    model_profile_path = os.path.join(profiles_dir, model_name + ".json")
    if os.path.isfile(model_profile_path):
        with open(model_profile_path, "r") as json_file:
            model_profile = json_file.read()
        #     model_graph: nx.MultiDiGraph = nx.node_link_graph(
        #         json.load(json_file)
        #     )

        # label_mapping = {
        #     node_name: NodeId(node_name) for node_name in model_graph.nodes
        # }
        # model_graph = nx.relabel_nodes(model_graph, label_mapping)
        return model_profile
    return None
