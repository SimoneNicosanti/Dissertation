import onnx
from Graph.Model.ModelGraph import ModelGraph
from Graph.Model.OnnxModelGraph import OnnxModelGraph
from Graph.Network.NetworkGraph import NetworkGraph
from GraphId import EdgeId, NodeId
from OptimizerClass import OptimizerClass
from Profiler.GraphProfile import EdgeInfo, GraphProfile, NodeInfo
from Profiler.ModelProfiler import ModelProfiler
from Profiler.OnnxModelProfiler import OnnxModelProfiler


def main():
    onnx_model = onnx.load_model("ResNet50.onnx")

    model_profile = OnnxModelProfiler().profile(
        onnx_model, {"args_0": (1, 3, 224, 224)}
    )

    model_graph: ModelGraph = OnnxModelGraph(onnx_model, model_profile)

    network_graph: NetworkGraph = NetworkGraph(prepare_network_profile())

    optimizer = OptimizerClass().optimize(model_graph, network_graph)


def prepare_network_profile():
    network_profile = GraphProfile()
    server_names = ["server_0", "server_1"]
    flops_list = [100, 50, 500, 250]
    energy_list = [2.49, 0.75, 0.1, 0.25]
    bandwidth_list = [1000.0, 100.0, 250.0, 100.0]

    for idx, server_name in enumerate(server_names):
        network_profile.put_node_profile(
            NodeId(server_name),
            NodeInfo(
                node_id=server_name,
                flops_per_sec=flops_list[idx % len(flops_list)],
                comp_energy_per_sec=energy_list[idx % len(energy_list)],
            ),
        )

    for idx_1, server_name in enumerate(server_names):
        for idx_2, other_server_name in enumerate(server_names):

            if server_name == other_server_name:
                bandwidth = bandwidth_list[0]
            else:
                bandwidth = bandwidth_list[(idx_1 + idx_2) % len(bandwidth_list)]
            network_profile.put_edge_profile(
                EdgeId(NodeId(server_name), NodeId(other_server_name)),
                EdgeInfo(
                    edge_id=EdgeId(NodeId(server_name), NodeId(other_server_name)),
                    bandwidth=bandwidth,
                ),
            )
    return network_profile


if __name__ == "__main__":
    main()
