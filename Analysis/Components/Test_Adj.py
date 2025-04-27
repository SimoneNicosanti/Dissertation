import itertools
import json

import matplotlib.pyplot as plt
import networkx as nx
import pulp
from networkx.drawing.nx_agraph import graphviz_layout

VAR_IDX = 0


def get_next_var_idx():
    global VAR_IDX
    VAR_IDX += 1
    return f"var_{VAR_IDX}"


def build_model_graph():
    with open("yolo11n-seg.json") as f:
        data = json.load(f)

    # Converte il JSON in grafo NetworkX
    model_graph = nx.readwrite.node_link_graph(data)
    return model_graph


def define_layer_comp_ass_vars(
    model_graph: nx.DiGraph, num_components: int
) -> dict[tuple, pulp.LpVariable]:
    var_dict = {}
    for node_id in model_graph.nodes:
        for comp_id in range(num_components):

            var_id = (node_id, comp_id)
            var = pulp.LpVariable(get_next_var_idx(), cat=pulp.LpBinary)

            var_dict[var_id] = var

    return var_dict


def define_comp_dependency_vars(num_components: int) -> dict[tuple, pulp.LpVariable]:
    var_dict = {}
    for comp_id in range(num_components):
        for other_comp_id in range(num_components):
            comp_edge_id = (comp_id, other_comp_id)

            var_id = comp_edge_id
            var = pulp.LpVariable(get_next_var_idx(), cat=pulp.LpBinary)

            var_dict[var_id] = var

    return var_dict


#################################################################################################################


def component_weight_cost(
    problem,
    model_graph: nx.DiGraph,
    num_components: int,
    layer_comp_ass_vars: dict[tuple],
):

    layer_max_flops = 0
    comp_cost_dict = {}
    for comp_id in range(num_components):
        comp_cost = 0
        for layer_id in model_graph.nodes:
            ass_var = layer_comp_ass_vars[(layer_id, comp_id)]
            layer_flops = model_graph.nodes[layer_id].get("flops", 0)
            comp_cost += ass_var * layer_flops

            layer_max_flops = max(layer_max_flops, layer_flops)

        comp_cost_dict[comp_id] = comp_cost

    comp_weight_cost = pulp.LpVariable(get_next_var_idx(), lowBound=0)
    for comp_id in range(num_components):
        problem += comp_weight_cost >= (comp_cost_dict[comp_id] / layer_max_flops)

    return comp_weight_cost


def cut_cost(
    problem,
    model_graph: nx.DiGraph,
    cut_edge_vars: dict[tuple],
):
    cut_cost = 0
    max_edge_cost = 0
    for edge_id in model_graph.edges:
        ass_var = cut_edge_vars[edge_id]

        edge_cost = model_graph.edges[edge_id].get("tot_tensor_size", 0)
        max_edge_cost = max(max_edge_cost, edge_cost)
        cut_cost += ass_var * edge_cost

    return cut_cost / max_edge_cost


def set_assignment_constraints(
    problem,
    model_graph: nx.DiGraph,
    num_components: int,
    layer_comp_ass_vars: dict[tuple],
    cut_edge_vars: dict[tuple],
):
    ## Layer --> Component
    for layer_id in model_graph.nodes:
        sum = 0
        for comp_id in range(num_components):
            sum += layer_comp_ass_vars[(layer_id, comp_id)]
        problem += sum == 1

    ## Sort of flow constraint
    for edge_id in model_graph.edges:
        for comp_id in range(num_components):
            problem += (
                layer_comp_ass_vars[(edge_id[1], comp_id)]
                - layer_comp_ass_vars[(edge_id[0], comp_id)]
                <= cut_edge_vars[edge_id]
            )

    pass


def set_acyclic_constraints(
    problem,
    num_components: int,
    comp_dependency_vars: dict[tuple, pulp.LpVariable],
    model_graph: nx.DiGraph,
    layer_comp_ass_vars: dict[tuple],
):
    for comp_id, next_comp_id in itertools.product(range(num_components), repeat=2):
        comp_edge_id = (comp_id, next_comp_id)
        if comp_id != next_comp_id:
            dep_var = comp_dependency_vars[comp_edge_id]
            for model_edge_id in model_graph.edges:
                problem += (
                    layer_comp_ass_vars[(model_edge_id[0], comp_id)]
                    + layer_comp_ass_vars[(model_edge_id[1], next_comp_id)]
                ) - 1 <= dep_var

    for comp_id, next_comp_id in itertools.product(range(num_components), repeat=2):
        if comp_id >= next_comp_id:
            dep_var = comp_dependency_vars[(comp_id, next_comp_id)]
            problem += dep_var == 0


def set_input_output_constraints(
    problem,
    model_graph: nx.DiGraph,
    num_components: int,
    layer_comp_ass_vars: dict[tuple, pulp.LpVariable],
):
    input_comp_id = 0
    output_comp_id = num_components - 1

    for layer_id in model_graph.nodes:
        if model_graph.nodes[layer_id].get("generator", False):
            layer_ass = layer_comp_ass_vars[(layer_id, input_comp_id)]
            problem += layer_ass == 1

    for layer_id in model_graph.nodes:
        if model_graph.nodes[layer_id].get("receiver", False):
            layer_ass = layer_comp_ass_vars[(layer_id, output_comp_id)]
            problem += layer_ass == 1

    input_comp_tot_nodes = 0
    for layer_id in model_graph.nodes:
        ass_var = layer_comp_ass_vars[(layer_id, input_comp_id)]
        input_comp_tot_nodes += ass_var
    problem += input_comp_tot_nodes == 1

    output_comp_tot_nodes = 0
    for layer_id in model_graph.nodes:
        ass_var = layer_comp_ass_vars[(layer_id, output_comp_id)]
        output_comp_tot_nodes += ass_var
    problem += output_comp_tot_nodes == 1


def main():
    num_components = 5 + 2
    model_graph: nx.DiGraph = build_model_graph()

    problem = pulp.LpProblem(name="Assignment", sense=pulp.LpMinimize)

    layer_comp_ass_vars = define_layer_comp_ass_vars(model_graph, num_components)

    comp_dependency_vars = define_comp_dependency_vars(num_components)

    cut_edge_vars = {
        edge_id: pulp.LpVariable(get_next_var_idx(), cat=pulp.LpBinary)
        for edge_id in model_graph.edges
    }

    set_assignment_constraints(
        problem,
        model_graph,
        num_components,
        layer_comp_ass_vars,
        cut_edge_vars,
    )

    set_acyclic_constraints(
        problem,
        num_components,
        comp_dependency_vars,
        model_graph,
        layer_comp_ass_vars,
    )

    set_input_output_constraints(
        problem,
        model_graph,
        num_components,
        layer_comp_ass_vars,
    )

    comp_cost = component_weight_cost(
        problem,
        model_graph,
        num_components,
        layer_comp_ass_vars,
    )

    edge_cut_cost = cut_cost(problem, model_graph, cut_edge_vars)

    problem += comp_cost + edge_cut_cost

    problem: pulp.LpProblem
    problem.solve(pulp.SCIP_PY())

    print("Status:", pulp.LpStatus[problem.status])

    valid_components = set()
    for key in layer_comp_ass_vars:
        if layer_comp_ass_vars[key].value() == 1:
            valid_components.add(key[1])

    dependency_graph = nx.DiGraph()
    for key in comp_dependency_vars:
        if (
            key[0] != key[1]
            and comp_dependency_vars[key].value() == 1
            and key[0] in valid_components
            and key[1] in valid_components
        ):
            dependency_graph.add_edge(key[0], key[1])

    pos = graphviz_layout(dependency_graph, prog="dot")
    nx.draw(dependency_graph, pos=pos, with_labels=True)
    plt.savefig("dependency_graph.png")


if __name__ == "__main__":
    main()
