import pulp
from Graph.ModelGraph import ModelGraph
from Graph.NetworkGraph import NetworkGraph
from Optimization import LatencyComputer
from Optimization.OptimizationKeys import EdgeAssKey, NodeAssKey

## TODO Check Normalization Min-Max: Per model or total


def find_energy_component(
    model_graphs: list[ModelGraph],
    network_graph: NetworkGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    requests_number: dict[str, int],
) -> pulp.LpAffineExpression:

    max_comp_energy = 0
    max_trans_energy = 0

    tot_comp_energy = 0
    tot_trans_energy = 0

    for curr_mod_graph in model_graphs:
        curr_comp_energy, curr_max_comp_energy = computation_energy(
            curr_mod_graph,
            network_graph,
            node_ass_vars,
            requests_number.get(curr_mod_graph.get_graph_name()),
        )
        max_comp_energy = max(max_comp_energy, curr_max_comp_energy)
        tot_comp_energy += curr_comp_energy

        curr_trans_energy, curr_max_comp_energy = transmission_energy(
            curr_mod_graph,
            network_graph,
            edge_ass_vars,
            requests_number.get(curr_mod_graph.get_graph_name()),
        )
        max_trans_energy = max(max_trans_energy, curr_max_comp_energy)
        tot_trans_energy += curr_trans_energy

    tot_comp_energy = tot_comp_energy  # / max_comp_energy
    tot_trans_energy = tot_trans_energy  # / max_trans_energy

    return tot_comp_energy, tot_trans_energy


def computation_energy(
    model_graph: ModelGraph,
    network_graph: NetworkGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    requests_number: int,
) -> tuple[pulp.LpAffineExpression, float]:
    sum_elems = []
    max_comp_enery = 0

    for net_node_id in network_graph.get_nodes_id():
        curr_net_node_comp_latency, curr_net_node_max_comp_latency = (
            LatencyComputer.node_computation_latency(
                model_graph, network_graph, node_ass_vars, net_node_id
            )
        )

        node_comp_energy = (
            curr_net_node_comp_latency
            * network_graph.get_node_info(net_node_id).get_comp_energy_per_sec()
        )
        max_node_comp_energy = (
            curr_net_node_max_comp_latency
            * network_graph.get_node_info(net_node_id).get_comp_energy_per_sec()
        )

        sum_elems.append(node_comp_energy)
        max_comp_enery = max(max_comp_enery, max_node_comp_energy)

    return (
        requests_number * pulp.lpSum(sum_elems) / max_comp_enery,
        requests_number * max_comp_enery,
    )


def transmission_energy(
    model_graph: ModelGraph,
    network_graph: NetworkGraph,
    edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    requests_number: int,
) -> tuple[pulp.LpAffineExpression, float]:
    sum_elems = []
    max_trans_enery = 0

    for net_node_id in network_graph.get_nodes_id():
        curr_net_node_trans_latency, curr_net_node_max_trans_latency = (
            LatencyComputer.node_transmission_latency(
                model_graph, network_graph, edge_ass_vars, net_node_id
            )
        )

        node_trans_energy = (
            curr_net_node_trans_latency
            * network_graph.get_node_info(net_node_id).get_trans_energy_per_sec()
        )
        max_node_trans_energy = (
            curr_net_node_max_trans_latency
            * network_graph.get_node_info(net_node_id).get_trans_energy_per_sec()
        )

        sum_elems.append(node_trans_energy)
        max_trans_enery = max(max_trans_enery, max_node_trans_energy)

    return (
        requests_number * pulp.lpSum(sum_elems) / max_trans_enery,
        requests_number * max_trans_enery,
    )
