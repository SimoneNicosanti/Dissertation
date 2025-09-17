import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from CommonIds.NodeId import NodeId
from CommonProfile.ExecutionProfile import (
    ModelExecutionProfile,
    ServerExecutionProfile,
    ServerExecutionProfilePool,
)
from CommonProfile.ModelProfile import ModelProfile, Regressor
from CommonProfile.NetworkInfo import NetworkEdgeInfo, NetworkNodeInfo
from CommonProfile.NetworkProfile import NetworkProfile

SKIP_LEN = 2
INPUT_SIZE = 4.5


def __build_model_branch(
    start_model_stats: dict,
    branch_size: int,
    random_generator: np.random.Generator,
    start_idx=0,
) -> nx.DiGraph:

    branch_model = nx.DiGraph(tensor_size_dict={})

    ## Default : Always start with a conv layer
    curr_type = "Conv"
    curr_idx = start_idx
    prev_node_id = None
    prev_node_out_size = None
    for _ in range(branch_size):
        layer_id = NodeId(f"{curr_type}_{curr_idx}")

        layer_type_flops_nums = start_model_stats["layer_types_info"][curr_type][
            "flops"
        ]
        layer_type_weight_sizes = start_model_stats["layer_types_info"][curr_type][
            "weights_size"
        ]
        layer_type_output_sizes = start_model_stats["layer_types_info"][curr_type][
            "outputs_size"
        ]

        flops = float(random_generator.choice(layer_type_flops_nums))
        weights_size = float(random_generator.choice(layer_type_weight_sizes))
        output_size = float(random_generator.choice(layer_type_output_sizes))

        branch_model.add_node(
            layer_id,
            flops=flops,
            weights_size=weights_size,
            idx=curr_idx,
            outputs_size=output_size,
            node_type=curr_type,
        )

        next_types = list(start_model_stats["edge_probabilities"][curr_type].keys())
        next_types_prob_array = list(
            start_model_stats["edge_probabilities"][curr_type].values()
        )
        next_type = str(random_generator.choice(next_types, p=next_types_prob_array))

        curr_type = next_type
        curr_idx += 1

        branch_model.graph["tensor_size_dict"][f"{layer_id.node_name}_Output"] = [
            layer_id.node_name,
            output_size,
        ]

        if prev_node_id is not None:

            branch_model.add_edge(
                prev_node_id,
                layer_id,
                tot_tensor_size=prev_node_out_size,
                tensor_name_list=[f"{prev_node_id.node_name}_Output"],
            )

        prev_node_id = layer_id
        prev_node_out_size = output_size

    return branch_model


def __merge_models(
    gen_model_graph: nx.DiGraph,
    main_branch: nx.DiGraph,
    curr_branch: nx.DiGraph,
    random_generator: np.random.Generator,
):
    if main_branch is not None:

        main_branch_nodes: list[NodeId] = list(nx.topological_sort(main_branch))
        curr_branch_nodes: list[NodeId] = list(nx.topological_sort(curr_branch))

        ## Get two random nodes from the main branch
        ## First node must be before the second node
        ## Use idx to check this
        main_branch_length = len(main_branch_nodes)
        ## Get Random Split Point in the first part of the branch
        split_node_id: NodeId = random_generator.choice(
            main_branch_nodes[: main_branch_length // 2]
        )

        ## Get Random Merge Point in the second part of the branch
        merge_node_id: NodeId = random_generator.choice(
            main_branch_nodes[main_branch_length // 2 + 1 :]
        )

        split_node_out_tensor = f"{split_node_id.node_name}_Output"
        split_node_out_tensor_size = main_branch.nodes[split_node_id]["outputs_size"]

        ## Adding fork edge
        curr_branch_start_node = curr_branch_nodes[0]

        curr_branch.add_node(
            split_node_id,
            flops=main_branch.nodes[split_node_id]["flops"],
            weights_size=main_branch.nodes[split_node_id]["weights_size"],
            idx=main_branch.nodes[split_node_id]["idx"],
            outputs_size=main_branch.nodes[split_node_id]["outputs_size"],
            node_type=main_branch.nodes[split_node_id]["node_type"],
        )
        curr_branch.add_edge(
            split_node_id,
            curr_branch_start_node,
            tot_tensor_size=split_node_out_tensor_size,
            tensor_name_list=[split_node_out_tensor],
        )

        curr_branch_last_node = curr_branch_nodes[-1]
        curr_branch_out_name = f"{curr_branch_last_node.node_name}_Output"
        curr_branch_out_size = curr_branch.nodes[curr_branch_last_node]["outputs_size"]

        curr_branch.add_node(
            merge_node_id,
            flops=main_branch.nodes[merge_node_id]["flops"],
            weights_size=main_branch.nodes[merge_node_id]["weights_size"],
            idx=main_branch.nodes[merge_node_id]["idx"],
            outputs_size=main_branch.nodes[merge_node_id]["outputs_size"],
            node_type=main_branch.nodes[merge_node_id]["node_type"],
        )

        ## Adding merge edge
        curr_branch.add_edge(
            curr_branch_last_node,
            merge_node_id,
            tot_tensor_size=curr_branch_out_size,
            tensor_name_list=[curr_branch_out_name],
        )

        # print(f"\t Fork >> {split_node_id} -> {curr_branch_start_node}")
        # print(f"\t Merge >> {curr_branch_last_node} -> {merge_node_id}")

    composed_gen_model_graph = nx.compose(gen_model_graph, curr_branch)

    for tensor_name, tensor_info in gen_model_graph.graph["tensor_size_dict"].items():
        composed_gen_model_graph.graph["tensor_size_dict"][tensor_name] = tensor_info

    for tensor_name, tensor_info in curr_branch.graph["tensor_size_dict"].items():
        composed_gen_model_graph.graph["tensor_size_dict"][tensor_name] = tensor_info

    return composed_gen_model_graph


def __add_skips_to_branch(
    curr_branch: nx.DiGraph, skip_prob: float, random_generator: np.random.Generator
):
    branch_nodes: list[NodeId] = list(nx.topological_sort(curr_branch))
    added_skips = 0
    for node_idx, node_id in enumerate(branch_nodes):
        if random_generator.random() < skip_prob:

            merge_node_idx = min(node_idx + SKIP_LEN, len(branch_nodes) - 1)
            merge_node_id = branch_nodes[merge_node_idx]
            if node_id == merge_node_id:
                continue

            tensor_out_name = f"{node_id.node_name}_Output"
            tensor_out_size = curr_branch.nodes[node_id]["outputs_size"]

            curr_branch.add_edge(
                node_id,
                merge_node_id,
                tot_tensor_size=tensor_out_size,
                tensor_name_list=[tensor_out_name],
            )

            added_skips += 1

            # print("Skip Connection >> ", node_id, " -> ", merge_node_id)

    if not nx.is_directed_acyclic_graph(curr_branch):
        raise Exception("Graph is not acyclic")

    # print(f"\t Added {added_skips} Skip Connections")


def __add_input_output_nodes(
    gen_model_graph: nx.DiGraph, main_branch: nx.DiGraph, curr_node_idx: int
):

    generator_id = NodeId("InputGenerator")
    receiver_id = NodeId("OutputReceiver")

    main_branch_nodes = list(nx.topological_sort(main_branch))
    main_branch_first_node = main_branch_nodes[0]
    main_branch_last_node = main_branch_nodes[-1]

    gen_model_graph.add_node(
        generator_id,
        flops=0,
        weights_size=0,
        idx=0,
        outputs_size=0,
        generator=True,
    )

    gen_model_graph.add_node(
        receiver_id,
        flops=0,
        weights_size=0,
        idx=curr_node_idx,
        outputs_size=0,
        receiver=True,
    )

    gen_model_graph.add_edge(
        generator_id,
        main_branch_first_node,
        tot_tensor_size=INPUT_SIZE,
        tensor_name_list=["InputGenerator_Output"],
    )

    gen_model_graph.add_edge(
        main_branch_last_node,
        receiver_id,
        tot_tensor_size=gen_model_graph.nodes[main_branch_last_node]["outputs_size"],
        tensor_name_list=[f"{main_branch_last_node}_Output"],
    )

    gen_model_graph.graph["tensor_size_dict"]["InputGenerator_Output"] = [
        generator_id.node_name,
        INPUT_SIZE,
    ]

    pass


def __build_model_graph(
    start_model: nx.DiGraph,
    branch_nodes: list[int],
    skip_prob: float = 0.0,
):
    random_generator = np.random.default_rng(seed=42)
    start_model_stats = __analyze_start_model(start_model)

    gen_model_graph = nx.DiGraph(name="GeneratedModel", tensor_size_dict={})
    main_branch = None
    curr_node_idx = 1
    for branch_idx, curr_branch_nodes in enumerate(branch_nodes):
        ## Generating Current Branch
        # print(f"Building Branch Num >> {branch_idx}")
        curr_branch = __build_model_branch(
            start_model_stats, curr_branch_nodes, random_generator, curr_node_idx
        )

        ## Adding skips to the current branch
        # print("Adding Skips")
        __add_skips_to_branch(curr_branch, skip_prob, random_generator)

        ## Merging Current Branch to the Main Model
        # print("Merging Models")
        gen_model_graph = __merge_models(
            gen_model_graph, main_branch, curr_branch, random_generator
        )

        ## Updating Main Branch
        if main_branch is None:
            main_branch = curr_branch

        curr_node_idx += curr_branch_nodes

    ## Add Input and Output Nodes to the Graph
    ## With their info as well (especially tensors info)
    __add_input_output_nodes(gen_model_graph, main_branch, curr_node_idx)

    return gen_model_graph


def __analyze_start_model(start_model: nx.DiGraph):

    edge_type_counters = {}
    edge_size_info = {}

    for edge_id in start_model.edges:
        ## Setting dest dict for this type of node
        start_layer, end_layer = edge_id

        ## Skipping input and output nodes
        if start_layer == "InputGenerator" or end_layer == "OutputReceiver":
            continue

        start_node_type = start_model.nodes[start_layer]["node_type"]
        edge_type_counters.setdefault(start_node_type, {})

        ## Setting dict for this kind of node with this kind of source
        end_node_type = start_model.nodes[end_layer]["node_type"]
        edge_type_counters[start_node_type].setdefault(end_node_type, 0)
        edge_type_counters[start_node_type][end_node_type] += 1

        ## Collecting possible sizes of tensors
        edge_type_key = (start_node_type, end_node_type)
        edge_size_info.setdefault(edge_type_key, [])
        edge_size_info[edge_type_key].append(
            start_model.edges[edge_id]["tot_tensor_size"]
        )

    for src_type in edge_type_counters.keys():
        dest_dict = edge_type_counters[src_type]
        tot_sum = sum(dest_dict.values())
        for dest_type in dest_dict.keys():
            edge_type_counters[src_type][dest_type] /= tot_sum

    layer_types_info = {}
    for node_id in start_model.nodes:
        if node_id == "InputGenerator" or node_id == "OutputReceiver":
            continue

        node_type = start_model.nodes[node_id]["node_type"]
        layer_types_info.setdefault(
            node_type, {"flops": [], "weights_size": [], "outputs_size": []}
        )
        layer_types_info[node_type]["flops"].append(start_model.nodes[node_id]["flops"])
        layer_types_info[node_type]["weights_size"].append(
            start_model.nodes[node_id]["weights_size"]
        )
        layer_types_info[node_type]["outputs_size"].append(
            start_model.nodes[node_id]["outputs_size"]
        )

    stats_dict = {}
    stats_dict["edge_probabilities"] = edge_type_counters
    stats_dict["edge_sizes"] = edge_size_info
    stats_dict["layer_types_info"] = layer_types_info

    return stats_dict


def build_model_profile(
    start_model: nx.DiGraph,
    branch_nodes: list[int],
    skip_prob: float = 0.0,
) -> ModelProfile:

    fake_profile = ModelProfile()

    generated_graph: nx.DiGraph = __build_model_graph(
        start_model, branch_nodes, skip_prob
    )
    fake_profile.set_model_graph(generated_graph)

    regressor = Regressor()
    fake_profile.set_regressor(regressor)

    return fake_profile


def build_network_profile(
    node_ids: list[int],
    node_avail_mem: dict[float],
    node_comp_energy_per_sec: dict[float],
    node_trans_energy_per_sec: dict[float],
    node_bandwidths: dict[dict[str, float]],
    node_latencies: dict[dict[str, float]],
) -> NetworkProfile:

    server_network = nx.DiGraph()

    ## Adding each node and its info
    for net_node_idx in node_ids:
        net_node_id = NodeId(net_node_idx)
        server_network.add_node(net_node_id)
        server_network.nodes[net_node_id][NetworkNodeInfo.AVAILABLE_MEMORY] = (
            node_avail_mem[net_node_idx]
        )
        server_network.nodes[net_node_id][NetworkNodeInfo.COMP_ENERGY_PER_SEC] = (
            node_comp_energy_per_sec[net_node_idx]
        )
        server_network.nodes[net_node_id][NetworkNodeInfo.TRANS_ENERGY_PER_SEC] = (
            node_trans_energy_per_sec[net_node_idx]
        )

        server_network.nodes[net_node_id][NetworkNodeInfo.TRANS_ENERGY_BASE] = 0
        server_network.nodes[net_node_id][NetworkNodeInfo.SELF_TRANS_ENERGY_PER_SEC] = 0
        server_network.nodes[net_node_id][NetworkNodeInfo.SELF_TRANS_ENERGY_BASE] = 0

        server_network.nodes[net_node_id][NetworkNodeInfo.IDX] = net_node_idx

    ## Adding each network edge and its info
    for src_node_idx in node_bandwidths:
        for dst_node_idx in node_bandwidths[src_node_idx]:
            src_node_id = NodeId(src_node_idx)
            dst_node_id = NodeId(dst_node_idx)

            bandwidth = node_bandwidths[src_node_idx][dst_node_idx]
            latency = node_latencies[src_node_idx][dst_node_idx]
            server_network.add_edge(src_node_id, dst_node_id)
            server_network.edges[src_node_id, dst_node_id][
                NetworkEdgeInfo.BANDWIDTH
            ] = bandwidth
            server_network.edges[src_node_id, dst_node_id][
                NetworkEdgeInfo.LATENCY
            ] = latency

    network_profile = NetworkProfile()
    network_profile.set_network_graph(server_network)

    return network_profile


## TODO >> Add for to cycle on servers
## Return Complete Object
def build_execution_profile(
    model_graph: nx.DiGraph, net_node_flops_dict: dict[int, float]
) -> ServerExecutionProfilePool:
    profile_pool = ServerExecutionProfilePool()

    for net_node_idx, net_node_flops in net_node_flops_dict.items():
        server_execution_profile = ServerExecutionProfile()

        execution_profile = ModelExecutionProfile()
        tot_sum = 0

        layer_id: NodeId
        for layer_id in model_graph.nodes:
            layer_flops = model_graph.nodes[layer_id]["flops"]
            net_node_exec_time = layer_flops / net_node_flops
            tot_sum += net_node_exec_time

            ## Maybe we can add some variability in here
            ## But it is not so necessary after all
            execution_profile.put_layer_execution_profile(
                NodeId(layer_id), net_node_exec_time, net_node_exec_time, False
            )

        execution_profile.put_layer_execution_profile(
            NodeId("WholeModel"), tot_sum, tot_sum, False
        )
        execution_profile.put_layer_execution_profile(
            NodeId("TotalSum"), tot_sum, tot_sum, False
        )

        server_execution_profile.put_model_execution_profile(
            model_graph.name, execution_profile
        )

        profile_pool.put_execution_profiles_for_server(
            NodeId(net_node_idx), server_execution_profile
        )

    return profile_pool
