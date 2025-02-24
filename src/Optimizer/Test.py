import onnx
from Graph.AssignmentGraph import AssignmentGraphInfo
from Graph.Graph import EdgeId, GraphInfo, NodeId
from Graph.ModelGraph import ModelGraph
from Graph.NetworkGraph import NetworkEdgeInfo, NetworkGraph, NetworkNodeInfo
from Optimization.AssignmentGraphBuilder import AssignmentGraphBuilder
from Optimization.OptimizationHandler import OptimizationHandler
from Optimization.SubGraphBuilder import SubGraphBuilder
from Partitioner.OnnxModelPartitioner import OnnxModelPartitioner
from Profiler.OnnxModelProfiler import OnnxModelProfiler


def main():

    node_id = NodeId("args_0")
    edge_id = EdgeId(node_id, node_id)

    onnx.utils.extract_model(
        "./models/ResNet50.onnx",
        "./models/ResNet50_sub.onnx",
        ["args_0"],
        ["resnet50/conv2_block1_3_bn/FusedBatchNormV3:0"],
    )

    model_graph: ModelGraph = OnnxModelProfiler(
        "./models/ResNet50_sub.onnx"
    ).profile_model({"args_0": (1, 3, 448, 448)})

    network_graph: NetworkGraph = prepare_network_profile()
    deployment_server = network_graph.get_nodes_id()[0]
    solved_prob_info = OptimizationHandler().optimize(
        model_graph, network_graph, deployment_server
    )

    sub_graph_builder = SubGraphBuilder(
        graph=model_graph, solved_problem_info=solved_prob_info
    )
    assignment_graph_builder = AssignmentGraphBuilder(
        graph=model_graph, solved_problem_info=solved_prob_info
    )
    assignment_graph = assignment_graph_builder.build_assignment_graph()

    print()
    print("Assignment Graph")
    for edge_id in assignment_graph.get_edges_id():
        print(edge_id)

    partitioner = OnnxModelPartitioner("./models/ResNet50_sub.onnx")
    partitioner.partition_model(assignment_graph)


def prepare_network_profile():
    graph = NetworkGraph()
    server_names = [
        "server_0",
        "server_1",
    ]
    flops_list = [5, 100]  # Edge, Fog, Cloud
    energy_list = [0.5, 1.0]  # Edge, Fog, Cloud
    bandwidth_list = [1, 20, 1]

    for idx, server_name in enumerate(server_names):
        node_id = NodeId(server_name)
        node_info = NetworkNodeInfo(
            {
                NetworkNodeInfo.Attributes.NET_NODE_FLOPS_PER_SEC: flops_list[
                    idx % len(flops_list)
                ],
                NetworkNodeInfo.Attributes.NET_NODE_COMP_ENERGY_PER_SEC: energy_list[
                    idx % len(energy_list)
                ],
                NetworkNodeInfo.Attributes.NET_NODE_TRANS_ENERGY_PER_SEC: energy_list[
                    idx % len(energy_list)
                ],
            }
        )
        graph.put_node(node_id, node_info)

    for idx_1, server_name in enumerate(server_names):
        for idx_2, other_server_name in enumerate(server_names):

            if server_name != other_server_name:
                bandwidth = 1000
                edge_info = NetworkEdgeInfo(
                    {NetworkEdgeInfo.Attributes.NET_EDGE_BANDWIDTH: bandwidth}
                )
            else:
                edge_info = GraphInfo({})

            edge_id = EdgeId(NodeId(server_name), NodeId(other_server_name))

            graph.put_edge(edge_id, edge_info)
    return graph


if __name__ == "__main__":
    main()
