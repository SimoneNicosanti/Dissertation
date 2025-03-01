import os

from Graph.ModelGraph import ModelGraph
from Graph.NetworkGraph import NetworkEdgeInfo, NetworkGraph, NetworkNodeInfo
from Optimization.AssignmentGraphBuilder import AssignmentGraphBuilder
from Optimization.OptimizationHandler import OptimizationHandler, OptimizationParams
from Partitioner.OnnxModelPartitioner import OnnxModelPartitioner
from Profiler.OnnxModelProfiler import OnnxModelProfiler


def main():

    # mod = YOLO("yolo11x-seg.pt")
    # mod.export(format="onnx")

    # return

    # # onnx.utils.extract_model(
    # #     "./models/ResNet50.onnx",
    # #     "./models/ResNet50_sub.onnx",
    # #     ["args_0"],
    # #     ["resnet50/conv2_block1_3_bn/FusedBatchNormV3:0"],
    # # )

    model_path = "./models/yolo11n-seg.onnx"
    model_graph_1: ModelGraph = OnnxModelProfiler(model_path).profile_model(
        {"args_0": (1, 3, 448, 448)}
    )

    model_path_2 = "./models/yolo11l-seg.onnx"
    model_graph_2: ModelGraph = OnnxModelProfiler(model_path_2).profile_model(
        {"args_0": (1, 3, 448, 448)}
    )

    model_path_3 = "./models/yolo11x-seg.onnx"
    model_graph_3: ModelGraph = OnnxModelProfiler(model_path_3).profile_model(
        {"args_0": (1, 3, 448, 448)}
    )

    graph_dict = {
        model_graph_1.get_graph_name(): model_graph_1,
        model_graph_2.get_graph_name(): model_graph_2,
        model_graph_3.get_graph_name(): model_graph_3,
    }

    # model_graph: ModelGraph = OnnxModelPartitioner(graph_dict).partition_model()

    opt_params = OptimizationParams(
        comp_lat_weight=1.0,
        trans_lat_weight=1.0,
        comp_en_weight=0,
        trans_en_weight=0,
        device_max_energy=1.0,
        requests_number={
            model_graph_1.get_graph_name(): 3,
            model_graph_2.get_graph_name(): 1,
            model_graph_3.get_graph_name(): 1,
        },
    )

    network_graph: NetworkGraph = prepare_network_profile()
    deployment_server = network_graph.get_nodes_id()[0]
    solved_prob_info_dict: dict[str] = OptimizationHandler().optimize(
        [model_graph_1, model_graph_2, model_graph_3],
        network_graph,
        deployment_server,
        opt_params=opt_params,
    )

    for graph_name in graph_dict.keys():
        solved_prob_info = solved_prob_info_dict[graph_name]

        assignment_graph_builder = AssignmentGraphBuilder(
            graph=graph_dict.get(graph_name), solved_problem_info=solved_prob_info
        )
        assignment_graph = assignment_graph_builder.build_assignment_graph()

        print()
        print("Assignment Graph")
        for edge_id in assignment_graph.get_edges_id():
            print(
                edge_id.first_node_id.node_name, ">", edge_id.second_node_id.node_name
            )

        partitioner = OnnxModelPartitioner(model_path)
        partitioner.partition_model(assignment_graph)


def prepare_network_profile():
    graph = NetworkGraph("NetworkGraph")
    server_names = ["0", "1"]
    flops_list = [5, 10, 10]  # Edge, Fog, Cloud
    energy_list = [0.5, 1.0]  # Edge, Fog, Cloud
    bandwidth_list = [1, 20, 1]

    for idx, server_name in enumerate(server_names):
        node_id = graph.build_node_id(server_name)
        node_info = NetworkNodeInfo(
            net_node_flops_per_sec=flops_list[idx % len(flops_list)],
            net_node_comp_energy_per_sec=energy_list[idx % len(energy_list)],
            net_node_trans_energy_per_sec=energy_list[idx % len(energy_list)],
            net_node_available_memory=1000 if idx == 0 else 100_000_000,
        )
        graph.put_node(node_id, node_info)

    for idx_1, server_name in enumerate(server_names):
        for idx_2, other_server_name in enumerate(server_names):

            if server_name != other_server_name:
                bandwidth = 5  ## Bandwidth in MB / s
                edge_info = NetworkEdgeInfo(net_edge_bandwidth=bandwidth)
            else:
                edge_info = NetworkEdgeInfo(net_edge_bandwidth=None)

            edge_id = graph.build_edge_id(server_name, other_server_name)

            graph.put_edge(edge_id, edge_info)
    return graph


if __name__ == "__main__":
    main()
