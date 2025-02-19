import onnx
from Graph.Graph import Edge, Node
from Graph.GraphId import EdgeId, NodeId
from Graph.GraphInfo import EdgeInfo, NodeInfo
from Graph.ModelGraph import ModelGraph
from Graph.NetworkGraph import NetworkGraph
from OptimizerClass import OptimizerClass
from Profiler.OnnxModelProfiler import OnnxModelProfiler


def main():
    onnx_model = onnx.load_model("./models/ResNet50.onnx")

    model_graph: ModelGraph = OnnxModelProfiler().profile_model(
        onnx_model, {"args_0": (1, 3, 224, 224)}
    )

    network_graph: NetworkGraph = prepare_network_profile()
    optimizer = OptimizerClass().optimize(model_graph, network_graph)


def prepare_network_profile():
    graph = NetworkGraph()
    server_names = [
        "server_0",
        "server_1",
        "server_2",
    ]
    flops_list = [100, 500, 1_000]  # Edge, Fog, Cloud
    energy_list = [0.3, 0.6, 1.0]  # Edge, Fog, Cloud
    bandwidth_list = [1e4, 100, 50, 25]

    for idx, server_name in enumerate(server_names):
        node_id = NodeId(server_name)
        node_info = NodeInfo(
            {
                NodeInfo.NET_NODE_FLOPS_PER_SEC: flops_list[idx % len(flops_list)],
                NodeInfo.NET_NODE_COMP_ENERGY_PER_SEC: energy_list[
                    idx % len(energy_list)
                ],
                NodeInfo.NET_NODE_TRANS_ENERGY_PER_SEC: energy_list[
                    idx % len(energy_list)
                ],
            }
        )
        node = Node(node_id, node_info)
        graph.put_node(node_id, node)

    for idx_1, server_name in enumerate(server_names):
        for idx_2, other_server_name in enumerate(server_names):

            if server_name == other_server_name:
                bandwidth = 1e4
            else:
                bandwidth = bandwidth_list[(idx_1 + idx_2) % len(bandwidth_list)]

            edge_id = EdgeId(NodeId(server_name), NodeId(other_server_name))
            edge_info = EdgeInfo({EdgeInfo.NET_EDGE_BANDWIDTH: bandwidth})
            edge = Edge(edge_id, edge_info)

            graph.put_edge(edge_id, edge)
    return graph


if __name__ == "__main__":
    main()
