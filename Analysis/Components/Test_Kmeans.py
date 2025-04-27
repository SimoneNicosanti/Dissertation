import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.cluster import OPTICS, AgglomerativeClustering


# Generare dati di esempio (3 cluster)
def build_model_graph():
    with open("yolo11n-seg.json") as f:
        data = json.load(f)

    # Converte il JSON in grafo NetworkX
    model_graph = nx.readwrite.node_link_graph(data)
    return model_graph


def main():
    model_graph = build_model_graph()

    nodes = sorted(list(model_graph.nodes), key=lambda x: model_graph.nodes[x]["idx"])
    n = len(nodes)
    max_flops = max([model_graph.nodes[i]["flops"] for i in model_graph.nodes])
    max_tensor_size = max(
        [model_graph.edges[i]["tot_tensor_size"] for i in model_graph.edges]
    )

    normalized_graph = nx.DiGraph()
    for edge_id in model_graph.edges:
        first_id = edge_id[0]

        edge_weight = (
            model_graph.edges[edge_id]["tot_tensor_size"] / max_tensor_size
            + model_graph.nodes[first_id]["flops"] / max_flops
        )

        normalized_graph.add_edge(
            edge_id[0],
            edge_id[1],
            weight=edge_weight,
        )

    # 2. Costruzione matrice delle distanze (basata sul peso minimo da i a j)
    dist_matrix = np.full((n, n), np.inf)
    for i, src in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(normalized_graph, src)
        for j, dst in enumerate(nodes):
            if dst in lengths:
                dist_matrix[i, j] = lengths[dst]

    max_dist = np.max(dist_matrix[np.isfinite(dist_matrix)])
    dist_matrix = dist_matrix / max_dist
    dist_matrix[dist_matrix == np.inf] = 200.0

    # clustering = AgglomerativeClustering(
    #     linkage="average",
    #     metric="precomputed",
    #     distance_threshold=0.25,
    #     n_clusters=None,
    # )

    clustering = OPTICS(metric="precomputed", min_samples=2)
    labels = clustering.fit_predict(dist_matrix)

    labels = clustering.fit_predict(dist_matrix)
    cluster_labels = {
        node: str(labels[i]) for i, node in enumerate(nodes)
    }  # etichette = numero cluster

    # Layout orientato (es. top-down)
    pos = graphviz_layout(model_graph, prog="dot")

    # Disegno del grafo
    plt.figure(figsize=(12, 10))
    nx.draw(
        model_graph,
        pos,
        with_labels=False,  # disattiva etichette default
        node_size=100,
        edge_color="gray",
        arrows=True,
        node_color="lightblue",
    )

    # Etichette: numeri dei cluster
    nx.draw_networkx_labels(
        model_graph, pos, labels=cluster_labels, font_color="black", font_size=10
    )

    plt.title("DAG con indice di cluster come etichetta")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
