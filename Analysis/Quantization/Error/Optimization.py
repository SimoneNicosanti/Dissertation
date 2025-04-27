import ast
import json

import joblib
import networkx as nx
import pulp

VAR_IDX = 0


def get_next_var_idx():
    global VAR_IDX
    VAR_IDX += 1
    return f"x_{VAR_IDX}"


def build_model_graph():
    with open("graph.json") as f:
        data = json.load(f)

    # Converte il JSON in grafo NetworkX
    model_graph = nx.readwrite.node_link_graph(data)
    return model_graph


def build_network_graph() -> nx.DiGraph:
    net_graph = nx.DiGraph()

    net_graph.add_node(0, flops=3 * 10**9)
    net_graph.add_node(1, flops=5 * 10**9)

    net_graph.add_edge(0, 0, bandwidth=1000)
    net_graph.add_edge(0, 1, bandwidth=10)
    net_graph.add_edge(1, 1, bandwidth=1000)
    net_graph.add_edge(1, 0, bandwidth=10)

    return net_graph


def add_product_constraint(
    problem: pulp.LpProblem,
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    layer_ass_vars: dict[tuple, pulp.LpVariable],
    layer_product_vars: dict[tuple, pulp.LpVariable],
    edge_ass_vars: dict[tuple, pulp.LpVariable],
    edge_product_vars: dict[tuple, pulp.LpVariable],
    quantization_vars: dict[tuple, pulp.LpVariable],
):

    ## Product definition for first part
    for layer_prod_key, layer_prod_var in layer_product_vars.items():
        layer_id, net_node_id = layer_prod_key

        problem += layer_prod_var <= layer_ass_vars[(layer_id, net_node_id)]
        problem += layer_prod_var <= quantization_vars[layer_id]
        problem += (
            layer_prod_var
            >= layer_ass_vars[(layer_id, net_node_id)] + quantization_vars[layer_id] - 1
        )

    ## Product definition for second part
    for edge_prod_key, edge_prod_var in edge_product_vars.items():
        edge_id, net_edge_id = edge_prod_key

        problem += edge_prod_var <= edge_ass_vars[(edge_id, net_edge_id)]
        problem += edge_prod_var <= quantization_vars[edge_id[0]]
        problem += (
            edge_prod_var
            >= edge_ass_vars[(edge_id, net_edge_id)] + quantization_vars[edge_id[0]] - 1
        )
    pass


def add_layer_assignment_constraint(
    problem: pulp.LpProblem,
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    layer_ass_vars: dict[tuple, pulp.LpVariable],
):

    for layer_id in model_graph.nodes:
        sum_var = 0
        for net_node_id in network_graph.nodes:
            sum_var += layer_ass_vars[(layer_id, net_node_id)]
        problem += sum_var == 1

    for layer_id in model_graph.nodes:
        if model_graph.nodes[layer_id].get("generator", False):
            layer_ass = layer_ass_vars[(layer_id, 0)]
            problem += layer_ass == 1
        if model_graph.nodes[layer_id].get("receiver", False):
            layer_ass = layer_ass_vars[(layer_id, 0)]
            problem += layer_ass == 1

    pass


def add_edge_assignment_constraint(
    problem: pulp.LpProblem,
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    edge_ass_vars: dict[tuple, pulp.LpVariable],
):
    for edge_id in model_graph.edges:
        sum_var = 0
        for net_edge_id in network_graph.edges:
            sum_var += edge_ass_vars[(edge_id, net_edge_id)]
        problem += sum_var == 1

    pass


def add_flow_constraint(
    problem: pulp.LpProblem,
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    layer_ass_vars: dict[tuple, pulp.LpVariable],
    edge_ass_vars: dict[tuple, pulp.LpVariable],
):
    ## Send flow
    for edge_id in model_graph.edges:
        for sending_net_node_id in network_graph.nodes:
            ass_var = layer_ass_vars[(edge_id[0], sending_net_node_id)]
            sum_vars = 0
            for receiving_net_node_id in network_graph.nodes:
                net_edge_id = (sending_net_node_id, receiving_net_node_id)
                sum_vars += edge_ass_vars[(edge_id, net_edge_id)]

            problem += ass_var == sum_vars

    ## Receive flow
    for edge_id in model_graph.edges:
        for receiving_net_node_id in network_graph.nodes:
            ass_var = layer_ass_vars[(edge_id[1], receiving_net_node_id)]
            sum_vars = 0
            for sending_net_node_id in network_graph.nodes:
                net_edge_id = (sending_net_node_id, receiving_net_node_id)
                sum_vars += edge_ass_vars[(edge_id, net_edge_id)]

            problem += ass_var == sum_vars

    pass


def add_quantization_constraint(
    problem: pulp.LpProblem,
    model_graph: nx.DiGraph,
    quantization_vars: dict[tuple, pulp.LpVariable],
):

    ## Only quantizable Layers will have not null quant var
    for layer_id in model_graph.nodes:
        if not model_graph.nodes[layer_id].get("quantizable", False):
            problem += quantization_vars[layer_id] == 0

    pass


def define_error_model(
    problem: pulp.LpProblem,
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    quantization_vars: dict[tuple, pulp.LpVariable],
):
    exist_var = pulp.LpVariable("exist", cat=pulp.LpBinary)

    quantizable_vars = {}
    for layer_id in model_graph.nodes:
        if model_graph.nodes[layer_id].get("quantizable", False):
            quantizable_vars[layer_id] = quantization_vars[layer_id]

    sum_vars = 0
    for _, quantizable_var in quantizable_vars.items():
        sum_vars += quantizable_var
        problem += exist_var >= quantizable_var
    problem += exist_var <= sum_vars

    with open("error_model.json", "r") as f:
        model_repr: dict = json.load(f)

    decoded_model_repr = {}
    decoded_model_repr["coef"] = {}
    decoded_model_repr["intercept"] = model_repr["intercept"]
    for key, value in model_repr["coef"].items():
        new_key = ast.literal_eval(key)
        decoded_model_repr["coef"][new_key] = value

    model_repr = decoded_model_repr
    regressor = 0
    for coef_key, coef_value in model_repr["coef"].items():

        if len(coef_key) == 1:
            layer_id = coef_key[0]
            regressor += coef_value * quantizable_vars[layer_id]
        else:
            quant_prod_var = pulp.LpVariable(get_next_var_idx(), cat=pulp.LpBinary)

            ## Defining constraint product <= single var
            quant_var_sum = 0
            for i in range(len(coef_key)):
                layer_id = coef_key[i]
                problem += quant_prod_var <= quantizable_vars[layer_id]
                quant_var_sum += quantizable_vars[layer_id]

            ## Defining constraint preduct >= sum - (k-1)
            problem += quant_prod_var >= quant_var_sum - (len(coef_key) - 1)

            regressor += coef_value * quant_prod_var

    regressor += model_repr["intercept"] * exist_var

    return regressor


def computation_time(
    problem: pulp.LpProblem,
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    layer_ass_vars: dict[tuple, pulp.LpVariable],
    layer_product_vars: dict[tuple, pulp.LpVariable],
):

    tot_time = 0
    for layer_id in model_graph.nodes:
        for net_node_id in network_graph.nodes:

            layer_comp_time = (
                model_graph.nodes[layer_id]["flops"]
                / network_graph.nodes[net_node_id]["flops"]
            )

            term = layer_comp_time * layer_ass_vars[(layer_id, net_node_id)]

            ## Quantized Case
            quant_layer_comp_time = layer_comp_time / 2
            term -= (layer_comp_time - quant_layer_comp_time) * layer_product_vars[
                (layer_id, net_node_id)
            ]

            tot_time += term

    return tot_time


def transfer_time(
    problem: pulp.LpProblem,
    model_graph: nx.DiGraph,
    network_graph: nx.DiGraph,
    edge_ass_vars: dict[tuple, pulp.LpVariable],
    edge_product_vars: dict[tuple, pulp.LpVariable],
):

    tot_time = 0
    for edge_id in model_graph.edges:
        for net_edge_id in network_graph.edges:

            tx_time = (
                model_graph.edges[edge_id]["tot_tensor_size"]
                / network_graph.edges[net_edge_id]["bandwidth"]
            )

            ## Not Quantized Case
            term = tx_time * edge_ass_vars[(edge_id, net_edge_id)]

            ## Quantized Case
            quant_tx_time = tx_time / 8
            term -= (tx_time - quant_tx_time) * edge_product_vars[
                (edge_id, net_edge_id)
            ]

            tot_time += term

    return tot_time
    pass


def main():

    problem = pulp.LpProblem(name="Assignment", sense=pulp.LpMinimize)
    model_graph = build_model_graph()
    network_graph = build_network_graph()

    layer_ass_vars = {}
    for layer_id in model_graph.nodes:
        for net_node_id in network_graph.nodes:
            layer_ass_vars[(layer_id, net_node_id)] = pulp.LpVariable(
                get_next_var_idx(), cat=pulp.LpBinary
            )

    edge_ass_vars = {}
    for edge_id in model_graph.edges:
        for net_edge_id in network_graph.edges:
            edge_ass_vars[(edge_id, net_edge_id)] = pulp.LpVariable(
                get_next_var_idx(), cat=pulp.LpBinary
            )

    quantization_vars = {}
    for layer_id in model_graph.nodes:
        quantization_vars[layer_id] = pulp.LpVariable(
            get_next_var_idx(), cat=pulp.LpBinary
        )

    layer_product_vars = {}
    for layer_id in model_graph.nodes:
        for net_node_id in network_graph.nodes:
            layer_product_vars[(layer_id, net_node_id)] = pulp.LpVariable(
                get_next_var_idx(), cat=pulp.LpBinary
            )

    edge_product_vars = {}
    for edge_id in model_graph.edges:
        for net_edge_id in network_graph.edges:
            edge_product_vars[(edge_id, net_edge_id)] = pulp.LpVariable(
                get_next_var_idx(), cat=pulp.LpBinary
            )

    add_product_constraint(
        problem,
        model_graph,
        network_graph,
        layer_ass_vars,
        layer_product_vars,
        edge_ass_vars,
        edge_product_vars,
        quantization_vars,
    )
    add_layer_assignment_constraint(problem, model_graph, network_graph, layer_ass_vars)
    add_edge_assignment_constraint(problem, model_graph, network_graph, edge_ass_vars)
    add_flow_constraint(
        problem, model_graph, network_graph, layer_ass_vars, edge_ass_vars
    )
    add_quantization_constraint(problem, model_graph, quantization_vars)

    regressor: pulp.LpAffineExpression = define_error_model(
        problem, model_graph, network_graph, quantization_vars
    )

    problem += regressor <= 0.3

    comp_time = computation_time(
        problem, model_graph, network_graph, layer_ass_vars, layer_product_vars
    )
    trans_time = transfer_time(
        problem, model_graph, network_graph, edge_ass_vars, edge_product_vars
    )

    problem += comp_time + trans_time

    problem.solve()

    tot_quantized = 0
    for _, var in quantization_vars.items():
        if var.varValue == 1:
            tot_quantized += var.varValue
    print("Tot Quantized >> ", tot_quantized)
    print("Regressor Value >> ", regressor.value())

    pass


if __name__ == "__main__":
    main()
