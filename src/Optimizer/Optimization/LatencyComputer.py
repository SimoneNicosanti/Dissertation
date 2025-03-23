## TODO Check Normalization Min-Max: Per model or total
import networkx as nx
import pulp

from CommonProfile.NodeId import NodeId
from CommonProfile.ModelInfo import ModelNodeInfo, ModelEdgeInfo
from Optimizer.Network.NetworkInfo import NetworkNodeInfo, NetworkEdgeInfo

from Optimizer.Optimization.OptimizationKeys import EdgeAssKey, NodeAssKey


def compute_latency_cost(
    model_graphs: list[nx.DiGraph],
    network_graph: nx.DiGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    requests_number: dict[str, int],
) -> pulp.LpAffineExpression:

    tot_latency_cost = 0

    total_requests = sum(requests_number.values())

    for model_graph in model_graphs:
        model_weight = requests_number.get(model_graph.graph["name"]) / total_requests

        model_comp_latency, max_model_comp_latency = compute_comp_latency_per_model(
            model_graph, network_graph, node_ass_vars
        )
        model_trans_latency, max_model_trans_latency = compute_trans_latency_per_model(
            model_graph, network_graph, edge_ass_vars
        )

        normalization_factor = max(max_model_comp_latency, max_model_trans_latency)

        tot_latency_cost += (
            model_weight
            * (model_comp_latency + model_trans_latency)
            / normalization_factor
        )

    return tot_latency_cost


def compute_comp_latency_per_model(
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
) -> tuple[pulp.LpAffineExpression, float]:
    tot_comp_latency = 0
    max_comp_latency = 0
    for net_node_id in network_graph.nodes:
        node_comp_latency_per_model, node_max_comp_latency_per_model = (
            compute_model_comp_latency_per_node(
                model_graph, network_graph, node_ass_vars, net_node_id
            )
        )
        tot_comp_latency += node_comp_latency_per_model
        max_comp_latency = max(max_comp_latency, node_max_comp_latency_per_model)

    return tot_comp_latency, max_comp_latency


def compute_trans_latency_per_model(
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
) -> tuple[pulp.LpAffineExpression, float]:
    tot_trans_latency = 0
    max_trans_latency = 0

    for net_node_id in network_graph.nodes:
        node_trans_latency_per_model, node_max_trans_latency_per_model = (
            compute_model_trans_latency_per_node(
                model_graph, network_graph, edge_ass_vars, net_node_id
            )
        )
        tot_trans_latency += node_trans_latency_per_model
        max_trans_latency = max(max_trans_latency, node_max_trans_latency_per_model)

    return tot_trans_latency, max_trans_latency


def compute_model_comp_latency_per_node(
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    net_node_id: NodeId,
) -> tuple[pulp.LpAffineExpression, float]:
    sum_elems = []
    max_comp_latency = 0
    for mod_node_id in model_graph.nodes:
        x_var_key = NodeAssKey(mod_node_id, net_node_id, model_graph.graph["name"])
        x_var = node_ass_vars[x_var_key]

        comp_time = __get_computation_time(
            model_graph.nodes[mod_node_id],
            network_graph.nodes[net_node_id],
        )
        max_comp_latency = max(max_comp_latency, comp_time)
        sum_elems.append(x_var * comp_time)

    return pulp.lpSum(sum_elems), max_comp_latency


def compute_model_trans_latency_per_node(
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    net_node_id: NodeId,
) -> tuple[pulp.LpAffineExpression, float]:
    sum_elems = []
    max_trans_latency = 0
    for mod_edge_id in model_graph.edges:
        for net_edge_id in network_graph.edges:
            if net_edge_id[0] == net_node_id:
                y_var_key = EdgeAssKey(
                    mod_edge_id,
                    net_edge_id,
                    model_graph.graph["name"],
                )
                y_var = edge_ass_vars[y_var_key]

                trans_time = __get_transmission_time(
                    model_graph.edges[mod_edge_id],
                    network_graph.edges[net_edge_id],
                    net_edge_id,
                )

                sum_elems.append(y_var * trans_time)

                max_trans_latency = max(
                    max_trans_latency,
                    trans_time,
                )
    return pulp.lpSum(sum_elems), max_trans_latency


def __get_transmission_time(
    mod_edge_info: dict,
    net_edge_info: dict,
    net_edge_id: tuple,
) -> float:
    ## Note --> Assuming Bandwidth in MB / s

    ## TODO Check This
    # if net_edge_id[0] == net_edge_id[1]:
    #     return 0

    transmission_time = (
        mod_edge_info[ModelEdgeInfo.TOT_TENSOR_SIZE]
        / net_edge_info[NetworkEdgeInfo.BANDWIDTH]
    )

    return transmission_time + net_edge_info[NetworkEdgeInfo.LATENCY]


def __get_computation_time(mod_node_info: dict, net_node_info: dict) -> float:
    return (
        mod_node_info[ModelNodeInfo.FLOPS]
        / net_node_info[NetworkNodeInfo.FLOPS_PER_SEC]
    )
