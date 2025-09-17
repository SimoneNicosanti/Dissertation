import networkx as nx
import pulp

from CommonIds.NodeId import NodeId
from CommonProfile.ExecutionProfile import ServerExecutionProfilePool
from CommonProfile.NetworkInfo import NetworkNodeInfo
from Optimizer.Optimization.LatencyComputer import LatencyComputer
from Optimizer.Optimization.OptimizationKeys import (
    NodeAssKey,
    TensorAssKey,
)


def compute_energy_cost(
    model_graphs: list[nx.MultiDiGraph],
    network_graph: nx.DiGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    tensor_ass_vars: dict[TensorAssKey, pulp.LpVariable],
    requests_number: dict[str, int],
    server_execution_profile_pool: ServerExecutionProfilePool,
    latency_computer: LatencyComputer,
) -> pulp.LpAffineExpression:

    tot_energy_cost = 0

    total_requests = sum(requests_number.values())

    for model_graph in model_graphs:
        model_weight = requests_number.get(model_graph.graph["name"]) / total_requests

        model_comp_energy, max_model_comp_energy = compute_comp_energy_per_model(
            model_graph,
            network_graph,
            node_ass_vars,
            server_execution_profile_pool,
            latency_computer,
        )
        model_trans_energy, max_model_trans_energy = compute_trans_energy_per_model(
            model_graph, network_graph, tensor_ass_vars, latency_computer
        )

        normalization_factor = 1  # max(max_model_comp_energy, max_model_trans_energy)

        tot_energy_cost += (
            model_weight
            * (model_comp_energy + model_trans_energy)
            / normalization_factor
        )

    return tot_energy_cost


def compute_comp_energy_per_model(
    model_graph: nx.MultiDiGraph,
    network_graph: nx.DiGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    server_execution_profile_pool: ServerExecutionProfilePool,
    latency_computer: LatencyComputer,
) -> tuple[pulp.LpAffineExpression, float]:
    tot_comp_energy = 0
    max_comp_energy = 0
    for net_node_id in network_graph.nodes:
        net_node_info: dict = network_graph.nodes[net_node_id]

        node_comp_latency_per_model, node_max_comp_latency_per_model = (
            latency_computer.compute_model_comp_latency_per_node(
                model_graph,
                network_graph,
                node_ass_vars,
                net_node_id,
                server_execution_profile_pool,
            )
        )

        node_comp_energy_per_model = (
            node_comp_latency_per_model * net_node_info["comp_energy_per_sec"]
        )
        node_max_comp_energy_per_model = (
            node_max_comp_latency_per_model * net_node_info["comp_energy_per_sec"]
        )

        tot_comp_energy += node_comp_energy_per_model
        max_comp_energy = max(max_comp_energy, node_max_comp_energy_per_model)

    return tot_comp_energy, max_comp_energy


def compute_trans_energy_per_model(
    model_graph: nx.MultiDiGraph,
    network_graph: nx.DiGraph,
    tensor_ass_vars: dict[TensorAssKey, pulp.LpVariable],
    latency_computer: LatencyComputer,
) -> tuple[pulp.LpAffineExpression, float]:
    tot_trans_energy = 0
    max_trans_energy = 0

    for net_node_id in network_graph.nodes:
        net_node_info: dict = network_graph.nodes[net_node_id]

        (
            node_trans_latency_per_model_same_dest,
            node_trans_latency_per_model_diff_dest,
            node_max_trans_latency_per_model,
        ) = latency_computer.compute_model_trans_latency_per_node(
            model_graph, network_graph, tensor_ass_vars, net_node_id
        )

        node_trans_energy_per_model_same_dest = (
            node_trans_latency_per_model_same_dest
            * net_node_info[NetworkNodeInfo.SELF_TRANS_ENERGY_PER_SEC]
            + net_node_info[NetworkNodeInfo.SELF_TRANS_ENERGY_BASE]
        )

        node_trans_energy_per_model_diff_dest = (
            node_trans_latency_per_model_diff_dest
            * net_node_info[NetworkNodeInfo.TRANS_ENERGY_PER_SEC]
            + net_node_info[NetworkNodeInfo.TRANS_ENERGY_BASE]
        )

        node_max_trans_energy_per_model = (
            node_max_trans_latency_per_model * net_node_info["trans_energy_per_sec"]
        )

        tot_trans_energy += (
            node_trans_energy_per_model_same_dest
            + node_trans_energy_per_model_diff_dest
        )
        max_trans_energy = max(max_trans_energy, node_max_trans_energy_per_model)

    return tot_trans_energy, max_trans_energy


def compute_energy_cost_per_net_node(
    model_graphs: list[nx.MultiDiGraph],
    network_graph: nx.DiGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    tensor_ass_vars: dict[TensorAssKey, pulp.LpVariable],
    requests_number: dict[str, int],
    net_node_id: NodeId,
    server_execution_profile_pool: ServerExecutionProfilePool,
    latency_computer: LatencyComputer,
) -> pulp.LpAffineExpression:

    net_node_energy = 0
    net_node_info: dict = network_graph.nodes[net_node_id]

    for model_graph in model_graphs:

        model_request_num = requests_number.get(model_graph.graph["name"])

        model_comp_latency, _ = latency_computer.compute_model_comp_latency_per_node(
            model_graph,
            network_graph,
            node_ass_vars,
            net_node_id,
            server_execution_profile_pool,
        )
        model_comp_energy = (
            model_comp_latency * net_node_info[NetworkNodeInfo.COMP_ENERGY_PER_SEC]
        )

        model_trans_latency_same_dest, model_trans_latency_diff_dest, _ = (
            latency_computer.compute_model_trans_latency_per_node(
                model_graph,
                network_graph,
                tensor_ass_vars,
                net_node_id,
            )
        )
        model_trans_energy = (
            model_trans_latency_diff_dest
            * net_node_info[NetworkNodeInfo.TRANS_ENERGY_PER_SEC]
        ) + net_node_info[NetworkNodeInfo.TRANS_ENERGY_BASE]

        net_node_energy += model_request_num * (model_comp_energy + model_trans_energy)

    return net_node_energy
