import pulp
from Optimizer.Graph.Graph import NodeId
from Optimizer.Graph.ModelGraph import ModelGraph
from Optimizer.Graph.NetworkGraph import NetworkGraph, NetworkNodeInfo
from Optimizer.Optimization import LatencyComputer
from Optimizer.Optimization.OptimizationKeys import EdgeAssKey, NodeAssKey

## TODO Check Normalization Min-Max: Per model or total


def compute_energy_cost(
    model_graphs: list[ModelGraph],
    network_graph: NetworkGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    requests_number: dict[str, int],
) -> pulp.LpAffineExpression:

    tot_energy_cost = 0

    total_requests = sum(requests_number.values())

    for model_graph in model_graphs:
        model_weight = (
            requests_number.get(model_graph.get_graph_name()) / total_requests
        )

        model_comp_energy, max_model_comp_energy = compute_comp_energy_per_model(
            model_graph, network_graph, node_ass_vars
        )
        model_trans_energy, max_model_trans_energy = compute_trans_energy_per_model(
            model_graph, network_graph, edge_ass_vars
        )

        normalization_factor = max(max_model_comp_energy, max_model_trans_energy)

        tot_energy_cost += (
            model_weight
            * (model_comp_energy + model_trans_energy)
            / normalization_factor
        )

    return tot_energy_cost


def compute_comp_energy_per_model(
    model_graph: ModelGraph,
    network_graph: NetworkGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
) -> tuple[pulp.LpAffineExpression, float]:
    tot_comp_energy = 0
    max_comp_energy = 0
    for net_node_id in network_graph.get_nodes_id():
        net_node_info: NetworkNodeInfo = network_graph.get_node_info(net_node_id)

        node_comp_latency_per_model, node_max_comp_latency_per_model = (
            LatencyComputer.compute_model_comp_latency_per_node(
                model_graph, network_graph, node_ass_vars, net_node_id
            )
        )

        node_comp_energy_per_model = (
            node_comp_latency_per_model * net_node_info.get_comp_energy_per_sec()
        )
        node_max_comp_energy_per_model = (
            node_max_comp_latency_per_model * net_node_info.get_comp_energy_per_sec()
        )

        tot_comp_energy += node_comp_energy_per_model
        max_comp_energy = max(max_comp_energy, node_max_comp_energy_per_model)

    return tot_comp_energy, max_comp_energy


def compute_trans_energy_per_model(
    model_graph: ModelGraph,
    network_graph: NetworkGraph,
    edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
) -> tuple[pulp.LpAffineExpression, float]:
    tot_trans_energy = 0
    max_trans_energy = 0

    for net_node_id in network_graph.get_nodes_id():
        net_node_info: NetworkNodeInfo = network_graph.get_node_info(net_node_id)

        node_trans_latency_per_model, node_max_trans_latency_per_model = (
            LatencyComputer.compute_model_trans_latency_per_node(
                model_graph, network_graph, edge_ass_vars, net_node_id
            )
        )

        node_trans_energy_per_model = (
            node_trans_latency_per_model * net_node_info.get_trans_energy_per_sec()
        )
        node_max_trans_energy_per_model = (
            node_max_trans_latency_per_model * net_node_info.get_trans_energy_per_sec()
        )

        tot_trans_energy += node_trans_energy_per_model
        max_trans_energy = max(max_trans_energy, node_max_trans_energy_per_model)

    return tot_trans_energy, max_trans_energy


def compute_energy_cost_per_net_node(
    model_graphs: list[ModelGraph],
    network_graph: NetworkGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    requests_number: dict[str, int],
    net_node_id: NodeId,
) -> pulp.LpAffineExpression:

    net_node_energy = 0
    net_node_info: NetworkNodeInfo = network_graph.get_node_info(net_node_id)

    for model_graph in model_graphs:

        model_request_num = requests_number.get(model_graph.get_graph_name())

        model_comp_latency, _ = LatencyComputer.compute_model_comp_latency_per_node(
            model_graph, network_graph, node_ass_vars, net_node_id
        )

        model_trans_latency, _ = LatencyComputer.compute_model_trans_latency_per_node(
            model_graph, network_graph, edge_ass_vars, net_node_id
        )

        model_comp_energy = model_comp_latency * net_node_info.get_comp_energy_per_sec()
        model_trans_energy = (
            model_trans_latency * net_node_info.get_trans_energy_per_sec()
        )

        net_node_energy += model_request_num * (model_comp_energy + model_trans_energy)

    return net_node_energy
