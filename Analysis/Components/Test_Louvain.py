import json

import community  # aka python-louvain
import matplotlib.pyplot as plt
import networkx as nx


def build_model_graph():
    with open("yolo11n-seg.json") as f:
        data = json.load(f)

    # Converte il JSON in grafo NetworkX
    model_graph = nx.readwrite.node_link_graph(data)
    return model_graph


# Crea un grafo
G = build_model_graph()

not_directed_graph = G.to_undirected()


# Applica Louvain
partition = community.best_partition(not_directed_graph, weight="tot_tensors_size")

# Stampa le comunità
for node, community_id in partition.items():
    print(f"Nodo {node}: Comunità {community_id}")

# Visualizza con colori
colors = [partition[n] for n in not_directed_graph.nodes()]
nx.draw(G, node_color=colors, cmap=plt.cm.Set3)
plt.show()
