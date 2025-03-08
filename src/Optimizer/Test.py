import time

from Graph import ConnectedComponents
from Graph.ModelGraph import ModelGraph
from Graph.NetworkGraph import NetworkEdgeInfo, NetworkGraph, NetworkNodeInfo
from Graph.SolvedModelGraph import SolvedModelGraph
from Optimization.OptimizationHandler import OptimizationHandler, OptimizationParams
from Partitioner.OnnxModelPartitioner import OnnxModelPartitioner
from Plan.Plan import Plan
from Profiler import ProfilePrinter
from Profiler.OnnxModelProfiler import OnnxModelProfiler


def main():

    # start = time.perf_counter_ns()
    # model_path_1 = "./models/yolo11x-seg_quant.onnx"
    # print("Profiling Model >> ", model_path_1)
    # model_graph_1: ModelGraph = OnnxModelProfiler(model_path_1).profile_model(
    #     {"args_0": (1, 3, 448, 448)}
    # )
    # end = time.perf_counter_ns()
    # print(f"Profiling Time for {model_path_1} >> {(end-start) / 1e9}")

    # ProfilePrinter.print_profile_csv(model_graph_1, "./yolo11x-seg_quant_prof.csv")

    start = time.perf_counter_ns()
    model_path_2 = "./models/yolo11n-seg.onnx"
    print("Profiling Model >> ", model_path_2)
    model_graph_2: ModelGraph = OnnxModelProfiler(model_path_2).profile_model(
        {"args_0": (1, 3, 448, 448)}
    )
    end = time.perf_counter_ns()
    print(f"Profiling Time for {model_path_2} >> {(end-start) / 1e9}")
    # ProfilePrinter.print_profile_csv(model_graph_2, "./yolo11n-seg_prof.csv")

    # start = time.perf_counter_ns()
    # model_path_3 = "./models/yolo11l-seg.onnx"
    # print("Profiling Model >> ", model_path_3)
    # model_graph_3: ModelGraph = OnnxModelProfiler(model_path_3).profile_model(
    #     {"args_0": (1, 3, 448, 448)}
    # )
    # end = time.perf_counter_ns()
    # print(f"Profiling Time for {model_path_3} >> {(end-start) / 1e9}")

    graph_dict = {
        # model_graph_1.get_graph_name(): model_graph_1,
        model_graph_2.get_graph_name(): model_graph_2,
        # model_graph_3.get_graph_name(): model_graph_3,
    }

    opt_params = OptimizationParams(
        latency_weight=1.0,
        energy_weight=0,
        device_max_energy=1.0,
        requests_number={key: 1 for key in graph_dict.keys()},
    )

    network_graph: NetworkGraph = prepare_network_profile()
    deployment_server = network_graph.get_nodes_id()[0]
    solved_graphs: list[SolvedModelGraph] = OptimizationHandler().optimize(
        list(graph_dict.values()),
        network_graph,
        deployment_server,
        opt_params=opt_params,
    )

    for solved_graph in solved_graphs:

        graph_name = solved_graph.get_graph_name()
        ConnectedComponents.ConnectedComponentsFinder.find_connected_components(
            solved_graph
        )

        model_plan = Plan(solved_graph)

        partitioner = OnnxModelPartitioner("./models/" + graph_name + ".onnx")
        partitioner.partition_model(
            model_plan, solved_graph.get_graph_name(), print_plan=True
        )


def prepare_network_profile(first_node_id: str = "0"):
    graph = NetworkGraph("NetworkGraph")
    server_names = [first_node_id, "1"]
    flops_list = [2.5 * 10**9, 5 * 10**9, 12.5]  # Edge, Fog, Cloud
    energy_list = [0.5, 1.0]  # Edge, Fog, Cloud
    bandwidth_list = [1, 20, 1]

    for idx, server_name in enumerate(server_names):
        node_id = graph.build_node_id(server_name)
        node_info = NetworkNodeInfo(
            net_node_flops_per_sec=flops_list[idx % len(flops_list)],
            net_node_comp_energy_per_sec=energy_list[idx % len(energy_list)],
            net_node_trans_energy_per_sec=energy_list[idx % len(energy_list)],
            net_node_available_memory=10 if idx == 0 else 100_000_000,
        )
        graph.put_node(node_id, node_info)

        if idx == 0:
            node_info.net_node_ip_address = "172.18.0.6"
            node_info.net_node_port = 50040

    for idx_1, server_name in enumerate(server_names):
        for idx_2, other_server_name in enumerate(server_names):

            if server_name != other_server_name:
                if (
                    server_name == "0"
                    and other_server_name == "1"
                    or server_name == "1"
                    and other_server_name == "0"
                ):
                    bandwidth = 2.5  ## Bandwidth in MB / s
                if (
                    server_name == "0"
                    and other_server_name == "2"
                    or server_name == "2"
                    and other_server_name == "0"
                ):
                    bandwidth = 1
                if (
                    server_name == "1"
                    and other_server_name == "2"
                    or server_name == "2"
                    and other_server_name == "1"
                ):
                    bandwidth = 10
                edge_info = NetworkEdgeInfo(net_edge_bandwidth=bandwidth)
            else:
                edge_info = NetworkEdgeInfo(net_edge_bandwidth=None)

            edge_id = graph.build_edge_id(server_name, other_server_name)

            graph.put_edge(edge_id, edge_info)
    return graph


if __name__ == "__main__":
    main()
