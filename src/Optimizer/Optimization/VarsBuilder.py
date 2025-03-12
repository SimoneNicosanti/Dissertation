import pulp
from Optimizer.Graph.Graph import EdgeId, NodeId
from Optimizer.Graph.ModelGraph import ModelGraph
from Optimizer.Graph.NetworkGraph import NetworkGraph
from Optimizer.Optimization.OptimizationKeys import EdgeAssKey, MemoryUseKey, NodeAssKey


def define_node_assignment_vars(
    model_graph: ModelGraph,
    network_graph: NetworkGraph,
) -> dict[NodeAssKey, pulp.LpVariable]:

    vars_table: dict[NodeAssKey, pulp.LpVariable] = {}
    for mod_node_id in model_graph.get_nodes_id():
        for net_node_id in network_graph.get_nodes_id():

            var_name: str = __build_assignment_var_name(
                mod_node_id, net_node_id, model_graph.get_graph_name()
            )
            lp_variable = pulp.LpVariable(var_name, cat=pulp.LpBinary)

            # problem.addVariable(lp_variable)
            table_key = NodeAssKey(
                mod_node_id, net_node_id, model_graph.get_graph_name()
            )
            vars_table[table_key] = lp_variable

    return vars_table


def define_edge_assignment_vars(
    model_graph: ModelGraph,
    network_graph: NetworkGraph,
) -> dict[EdgeAssKey, pulp.LpVariable]:
    vars_table: dict[EdgeAssKey, pulp.LpVariable] = {}
    for mod_edge_id in model_graph.get_edges_id():
        for net_edge_id in network_graph.get_edges_id():

            var_name: str = __build_edge_var_name(
                mod_edge_id, net_edge_id, model_graph.get_graph_name()
            )
            lp_variable = pulp.LpVariable(var_name, cat=pulp.LpBinary)

            table_key = EdgeAssKey(
                mod_edge_id, net_edge_id, model_graph.get_graph_name()
            )
            vars_table[table_key] = lp_variable

    return vars_table


def define_memory_use_vars(
    network_graph: NetworkGraph, graph_name: str
) -> dict[NodeId, pulp.LpVariable]:
    vars_table: dict[NodeId, pulp.LpVariable] = {}
    for net_node_id in network_graph.get_nodes_id():
        var_name: str = "mem_(net_node_{})({})".format(net_node_id, graph_name)
        var_key = MemoryUseKey(graph_name, net_node_id)
        vars_table[var_key] = pulp.LpVariable(
            var_name, cat=pulp.LpContinuous, lowBound=0
        )

    return vars_table


def __build_assignment_var_name(modelNode: NodeId, networkNode: NodeId, modelName: str):
    return "x_(mod_node_{})(net_node_{})({})".format(modelNode, networkNode, modelName)


def __build_edge_var_name(modelEdge: EdgeId, networkEdge: EdgeId, modelName: str):
    return "y_(mod_node_{})(net_node_{})({})".format(modelEdge, networkEdge, modelName)
