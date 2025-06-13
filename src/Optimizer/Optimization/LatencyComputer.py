## TODO Check Normalization Min-Max: Per model or total
import networkx as nx
import pulp

from CommonIds.NodeId import NodeId
from CommonProfile.ExecutionProfile import (
    ModelExecutionProfile,
    ServerExecutionProfilePool,
)
from CommonProfile.ModelInfo import ModelEdgeInfo, ModelNodeInfo
from CommonProfile.NetworkInfo import NetworkEdgeInfo, NetworkNodeInfo
from Optimizer.Optimization.OptimizationKeys import EdgeAssKey, NodeAssKey


def compute_latency_cost(
    model_graphs: list[nx.DiGraph],
    network_graph: nx.DiGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    requests_number: dict[str, int],
    server_execution_profile_pool: ServerExecutionProfilePool,
) -> pulp.LpAffineExpression:

    tot_latency_cost = 0

    total_requests = sum(requests_number.values())

    for model_graph in model_graphs:
        model_weight = requests_number.get(model_graph.graph["name"]) / total_requests

        model_comp_latency, max_model_comp_latency = compute_comp_latency_per_model(
            model_graph, network_graph, node_ass_vars, server_execution_profile_pool
        )
        model_trans_latency, max_model_trans_latency = compute_trans_latency_per_model(
            model_graph, network_graph, edge_ass_vars
        )

        normalization_factor = 1  # max(max_model_comp_latency, max_model_trans_latency)

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
    server_execution_profile_pool: ServerExecutionProfilePool,
) -> tuple[pulp.LpAffineExpression, float]:
    tot_comp_latency = 0
    max_comp_latency = 0
    for net_node_id in network_graph.nodes:
        node_comp_latency_per_model, node_max_comp_latency_per_model = (
            compute_model_comp_latency_per_node(
                model_graph,
                network_graph,
                node_ass_vars,
                net_node_id,
                server_execution_profile_pool,
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
        (
            node_trans_latency_per_model_same_dest,
            node_trans_latency_per_model_diff_dest,
            node_max_trans_latency_per_model,
        ) = compute_model_trans_latency_per_node(
            model_graph, network_graph, edge_ass_vars, net_node_id
        )
        tot_trans_latency += (
            node_trans_latency_per_model_same_dest
            + node_trans_latency_per_model_diff_dest
        )
        max_trans_latency = max(max_trans_latency, node_max_trans_latency_per_model)

    return tot_trans_latency, max_trans_latency


def compute_model_comp_latency_per_node(
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    net_node_id: NodeId,
    server_execution_profile_pool: ServerExecutionProfilePool,
) -> tuple[pulp.LpAffineExpression, float]:
    sum_elems = []
    max_comp_latency = 0
    for mod_node_id in model_graph.nodes:
        server_model_execution_profile: ModelExecutionProfile = (
            server_execution_profile_pool.get_execution_profiles_for_server(
                net_node_id
            ).get_model_execution_profile(model_graph.graph["name"])
        )

        time_expr, layer_max_comp_time = __get_computation_time(
            model_graph,
            mod_node_id,
            net_node_id,
            server_model_execution_profile,
            node_ass_vars,
        )
        max_comp_latency = max(max_comp_latency, layer_max_comp_time)
        sum_elems.append(time_expr)

    return pulp.lpSum(sum_elems), max_comp_latency


def compute_model_trans_latency_per_node(
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    net_node_id: NodeId,
) -> tuple[pulp.LpAffineExpression, pulp.LpAffineExpression, float]:
    sum_elems_same_dest = []
    sum_elems_diff_dest = []
    max_trans_latency = 0
    for mod_edge_id in model_graph.edges:
        for net_edge_id in network_graph.edges:
            if net_edge_id[0] == net_node_id:
                trans_time_expr, layer_max_trans_time = __get_transmission_time(
                    model_graph,
                    network_graph,
                    mod_edge_id,
                    net_edge_id,
                    edge_ass_vars,
                )

                if net_edge_id[1] == net_node_id:
                    sum_elems_same_dest.append(trans_time_expr)
                else:
                    sum_elems_diff_dest.append(trans_time_expr)

                max_trans_latency = max(
                    max_trans_latency,
                    layer_max_trans_time,
                )

    return (
        pulp.lpSum(sum_elems_same_dest),
        pulp.lpSum(sum_elems_diff_dest),
        max_trans_latency,
    )


def __get_transmission_time(
    model_graph: nx.MultiDiGraph,
    network_graph: nx.DiGraph,
    mod_edge_id: tuple,
    net_edge_id: tuple,
    edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
) -> float:
    ## Note --> Assuming Bandwidth in MB / s

    not_quant_ass_key = EdgeAssKey(mod_edge_id, net_edge_id, model_graph.graph["name"])
    not_quant_tx_time = model_graph.edges[mod_edge_id][
        ModelEdgeInfo.TOT_TENSOR_SIZE
    ] / (network_graph.edges[net_edge_id][NetworkEdgeInfo.BANDWIDTH])
    trans_expr = not_quant_tx_time * edge_ass_vars[not_quant_ass_key]

    if model_graph.nodes[mod_edge_id[0]].get(ModelNodeInfo.QUANTIZABLE, False):
        quant_tx_time = not_quant_tx_time / 8

        quant_ass_key = EdgeAssKey(
            mod_edge_id, net_edge_id, model_graph.graph["name"], True
        )

        trans_expr = (
            not_quant_tx_time * edge_ass_vars[not_quant_ass_key]
            - (not_quant_tx_time - quant_tx_time) * edge_ass_vars[quant_ass_key]
        )

    ## We add the latency of this link only if the edge is actually mapped on this link
    latency = network_graph.edges[net_edge_id][NetworkEdgeInfo.LATENCY]
    trans_expr = trans_expr + latency * edge_ass_vars[not_quant_ass_key]

    return trans_expr, not_quant_tx_time + latency


def __get_computation_time(
    model_graph: nx.DiGraph,
    mod_node_id: NodeId,
    net_node_id: NodeId,
    model_execution_profile: ModelExecutionProfile,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
) -> float:

    not_quant_ass_key = NodeAssKey(mod_node_id, net_node_id, model_graph.graph["name"])
    not_quant_time = model_execution_profile.get_not_quantized_layer_time(mod_node_id)
    time_expr = not_quant_time * node_ass_vars[not_quant_ass_key]

    max_comp_time = not_quant_time

    if model_graph.nodes[mod_node_id].get(ModelNodeInfo.QUANTIZABLE, False):
        quant_time = model_execution_profile.get_quantized_layer_time(mod_node_id)
        quant_ass_key = NodeAssKey(
            mod_node_id, net_node_id, model_graph.graph["name"], True
        )

        time_expr = (
            not_quant_time * node_ass_vars[not_quant_ass_key]
            - (not_quant_time - quant_time) * node_ass_vars[quant_ass_key]
        )

        max_comp_time = max(max_comp_time, quant_time)

    return time_expr, max_comp_time
