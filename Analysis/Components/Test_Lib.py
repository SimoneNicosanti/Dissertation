import json

import networkx as nx
import pymetis


def get_next_var_idx():
    global VAR_IDX
    VAR_IDX += 1
    return f"var_{VAR_IDX}"


def build_model_graph():
    with open("yolo11n-seg.json") as f:
        data = json.load(f)

    # Converte il JSON in grafo NetworkX
    model_graph = nx.readwrite.node_link_graph(data)
    return model_graph


model_graph = build_model_graph()
adj = {}

for edge in model_graph.edges:
    pass


pymetis.part_graph()
