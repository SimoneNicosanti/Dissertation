import networkx as nx
import pulp

from Optimizer.Graph.Graph import ModelNodeInfo, NetworkNodeInfo, NodeId
from Optimizer.Optimization.OptimizationKeys import EdgeAssKey, MemoryUseKey, NodeAssKey


def define_node_assignment_vars(
    model_graph: nx.MultiDiGraph,
    network_graph: nx.DiGraph,
) -> dict[NodeAssKey, pulp.LpVariable]:

    vars_table: dict[NodeAssKey, pulp.LpVariable] = {}
    for mod_node_id in list(model_graph.nodes):
        for net_node_id in list(network_graph.nodes):

            mod_node_idx = model_graph.nodes[mod_node_id][ModelNodeInfo.IDX]
            net_node_idx = network_graph.nodes[net_node_id][NetworkNodeInfo.IDX]
            var_name: str = __build_assignment_var_name(
                mod_node_idx, net_node_idx, model_graph.graph["name"]
            )
            lp_variable = pulp.LpVariable(var_name, cat=pulp.LpBinary)

            # problem.addVariable(lp_variable)
            table_key = NodeAssKey(mod_node_id, net_node_id, model_graph.graph["name"])
            vars_table[table_key] = lp_variable

    return vars_table


def define_edge_assignment_vars(
    model_graph: nx.MultiDiGraph,
    network_graph: nx.DiGraph,
) -> dict[EdgeAssKey, pulp.LpVariable]:
    vars_table: dict[EdgeAssKey, pulp.LpVariable] = {}
    for mod_edge_id in list(model_graph.edges):
        for net_edge_id in list(network_graph.edges):
            var_name: str = __build_edge_var_name(
                mod_edge_id, net_edge_id, model_graph, network_graph
            )
            lp_variable = pulp.LpVariable(var_name, cat=pulp.LpBinary)

            table_key = EdgeAssKey(mod_edge_id, net_edge_id, model_graph.graph["name"])
            vars_table[table_key] = lp_variable

    return vars_table


def define_memory_use_vars(
    network_graph: nx.DiGraph, graph_name: str
) -> dict[MemoryUseKey, pulp.LpVariable]:
    vars_table: dict[MemoryUseKey, pulp.LpVariable] = {}
    for net_node_id in list(network_graph.nodes):  # network_graph.nodes:
        var_name: str = "mem_(net_node_{})({})".format(
            network_graph.nodes[net_node_id][NetworkNodeInfo.IDX], graph_name
        )
        var_key = MemoryUseKey(graph_name, net_node_id)
        vars_table[var_key] = pulp.LpVariable(
            var_name, cat=pulp.LpContinuous, lowBound=0
        )

    return vars_table


def __build_assignment_var_name(
    model_node_idx: int, net_node_idx: int, model_name: str
):
    return "x_(mod_node_{})(net_node_{})({})".format(
        model_node_idx, net_node_idx, model_name
    )


def __build_edge_var_name(
    mod_edge_id: tuple[NodeId, NodeId],
    net_edge_id: tuple[NodeId, NodeId],
    model_graph: nx.MultiDiGraph,
    network_graph: nx.DiGraph,
):
    mod_src_node, mod_dst_node = mod_edge_id
    mod_src_idx, mod_dst_idx = (
        model_graph.nodes[mod_src_node][ModelNodeInfo.IDX],
        model_graph.nodes[mod_dst_node][ModelNodeInfo.IDX],
    )

    net_src_node, net_dst_node = net_edge_id
    net_src_idx, net_dst_idx = (
        network_graph.nodes[net_src_node][NetworkNodeInfo.IDX],
        network_graph.nodes[net_dst_node][NetworkNodeInfo.IDX],
    )

    model_name = model_graph.graph["name"]
    mod_edge_str = "({}_{})".format(mod_src_idx, mod_dst_idx)
    net_edge_str = "({}_{})".format(net_src_idx, net_dst_idx)
    return "y_(mod_node_{})(net_node_{})({})".format(
        mod_edge_str, net_edge_str, model_name
    )
