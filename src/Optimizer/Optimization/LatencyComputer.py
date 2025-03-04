import pulp
from Graph.Graph import EdgeId, NodeId
from Graph.ModelGraph import ModelEdgeInfo, ModelGraph, ModelNodeInfo
from Graph.NetworkGraph import NetworkEdgeInfo, NetworkGraph, NetworkNodeInfo
from Optimization.OptimizationKeys import EdgeAssKey, ExpressionKey, NodeAssKey

## TODO Check Normalization Min-Max: Per model or total


def find_latency_component(
    model_graphs: list[ModelGraph],
    network_graph: NetworkGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    requests_number: dict[str, int],
) -> pulp.LpAffineExpression:

    max_comp_latency = 0
    max_trans_latency = 0

    tot_comp_latency = 0
    tot_trans_latency = 0

    for curr_mod_graph in model_graphs:
        curr_comp_latency, curr_max_comp_latency = computation_latency(
            curr_mod_graph,
            network_graph,
            node_ass_vars,
            requests_number.get(curr_mod_graph.get_graph_name()),
        )
        max_comp_latency = max(max_comp_latency, curr_max_comp_latency)
        tot_comp_latency += curr_comp_latency

        curr_trans_latency, curr_max_comp_latency = transmission_latency(
            curr_mod_graph,
            network_graph,
            edge_ass_vars,
            requests_number.get(curr_mod_graph.get_graph_name()),
        )
        max_trans_latency = max(max_trans_latency, curr_max_comp_latency)
        tot_trans_latency += curr_trans_latency

    tot_trans_latency = tot_trans_latency  # / max_trans_latency
    tot_comp_latency = tot_comp_latency  # / max_comp_latency

    return tot_comp_latency, tot_trans_latency


def computation_latency(
    model_graph: ModelGraph,
    network_graph: NetworkGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    requests_number: int,
) -> tuple[pulp.LpAffineExpression, float]:

    sum_elems = []
    max_comp_latency = 0
    for net_node_id in network_graph.get_nodes_id():
        curr_net_node_comp_latency, curr_net_node_max_comp_latency = (
            node_computation_latency(
                model_graph, network_graph, node_ass_vars, net_node_id
            )
        )
        sum_elems.append(curr_net_node_comp_latency)
        max_comp_latency = max(max_comp_latency, curr_net_node_max_comp_latency)

    return (
        requests_number * pulp.lpSum(sum_elems) / max_comp_latency,
        requests_number * max_comp_latency,
    )


def node_computation_latency(
    model_graph: ModelGraph,
    network_graph: NetworkGraph,
    node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    net_node_id: NodeId,
) -> tuple[pulp.LpAffineExpression, float]:
    sum_elems = []
    max_comp_latency = 0
    for mod_node_id in model_graph.get_nodes_id():
        x_var_key = NodeAssKey(mod_node_id, net_node_id, model_graph.get_graph_name())
        x_var = node_ass_vars[x_var_key]

        comp_time = __get_computation_time(
            model_graph.get_node_info(mod_node_id),
            network_graph.get_node_info(net_node_id),
        )
        max_comp_latency = max(max_comp_latency, comp_time)
        sum_elems.append(x_var * comp_time)

    return pulp.lpSum(sum_elems), max_comp_latency


def transmission_latency(
    model_graph: ModelGraph,
    network_graph: NetworkGraph,
    edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    requests_number: int,
) -> tuple[pulp.LpAffineExpression, float]:
    sum_elems = []
    max_trans_latency = 0
    for net_node_id in network_graph.get_nodes_id():
        curr_net_node_trans_latency, curr_max_trans_latency = node_transmission_latency(
            model_graph, network_graph, edge_ass_vars, net_node_id
        )
        sum_elems.append(curr_net_node_trans_latency)
        max_trans_latency = max(
            max_trans_latency,
            curr_max_trans_latency,
        )

    return (
        requests_number * pulp.lpSum(sum_elems) / max_trans_latency,
        requests_number * max_trans_latency,
    )


def node_transmission_latency(
    model_graph: ModelGraph,
    network_graph: NetworkGraph,
    edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    net_node_id: NodeId,
) -> tuple[pulp.LpAffineExpression, float]:
    sum_elems = []
    max_trans_latency = 0
    for mod_edge_id in model_graph.get_edges_id():
        for net_edge_id in network_graph.get_edges_id():
            if net_edge_id.first_node_id == net_node_id:
                y_var_key = EdgeAssKey(
                    mod_edge_id,
                    net_edge_id,
                    model_graph.get_graph_name(),
                )
                y_var = edge_ass_vars[y_var_key]

                trans_time = __get_transmission_time(
                    model_graph.get_edge_info(mod_edge_id),
                    network_graph.get_edge_info(net_edge_id),
                    net_edge_id,
                )

                sum_elems.append(y_var * trans_time)

                max_trans_latency = max(
                    max_trans_latency,
                    trans_time,
                )
    return pulp.lpSum(sum_elems), max_trans_latency


def __get_transmission_time(
    mod_edge_info: ModelEdgeInfo,
    net_edge_info: NetworkEdgeInfo,
    net_edge_id: EdgeId,
) -> float:
    ## Note --> Assuming Bandwidth in Byte / s

    if net_edge_id.first_node_id == net_edge_id.second_node_id:
        return 0

    return mod_edge_info.get_model_edge_data_size() / net_edge_info.get_edge_bandwidth()


def __get_computation_time(
    mod_node_info: ModelNodeInfo, net_node_info: NetworkNodeInfo
) -> float:
    return mod_node_info.get_node_flops() / net_node_info.get_flops_per_sec()
