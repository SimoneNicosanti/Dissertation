import itertools
import json

import networkx as nx
import pulp

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


def build_network_graph():
    network_graph = nx.DiGraph()
    for i, j in itertools.product(range(2), repeat=2):
        network_graph.add_edge(i, j)

    return network_graph


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


def define_comp_server_ass_vars(
    num_components: int, network_graph: int
) -> dict[tuple, pulp.LpVariable]:
    var_dict = {}
    for comp_id in range(num_components):
        for server_id in network_graph.nodes:
            var_id = (comp_id, server_id)
            var = pulp.LpVariable(get_next_var_idx(), cat=pulp.LpBinary)

            var_dict[var_id] = var

    return var_dict


def define_model_edge_comp_edge_ass_vars(
    model_graph: nx.DiGraph, num_components: int
) -> dict[tuple, pulp.LpVariable]:
    var_dict = {}
    for model_edge_id in model_graph.edges:
        for comp_id in range(num_components):
            for other_comp_id in range(num_components):
                comp_edge_id = (comp_id, other_comp_id)

                var_id = (model_edge_id, comp_edge_id)
                var = pulp.LpVariable(get_next_var_idx(), cat=pulp.LpBinary)

                var_dict[var_id] = var

    return var_dict


def define_model_node_comp_node_ass_vars(
    num_components: int, network_graph: nx.DiGraph
) -> dict[tuple, pulp.LpVariable]:
    var_dict = {}
    for comp_id in range(num_components):
        for other_comp_id in range(num_components):
            comp_edge_id = (comp_id, other_comp_id)
            for net_edge_id in network_graph.edges:
                var_id = (comp_edge_id, net_edge_id)
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


def component_computation_time(
    problem,
    model_graph: nx.DiGraph,
    num_components: int,
    network_graph: nx.DiGraph,
    layer_comp_ass_vars: dict[tuple, pulp.LpVariable],
    comp_server_ass_vars: dict[tuple, pulp.LpVariable],
):
    time_dict = {}
    for comp_id in range(num_components):
        component_time = 0
        for server_id in network_graph.nodes:
            comp_ass_var = comp_server_ass_vars[(comp_id, server_id)]

            for layer_id in model_graph.nodes:
                layer_ass_var = layer_comp_ass_vars[(layer_id, comp_id)]

                product_var = pulp.LpVariable(get_next_var_idx(), cat=pulp.LpBinary)

                problem += product_var <= comp_ass_var
                problem += product_var <= layer_ass_var
                problem += product_var >= comp_ass_var + layer_ass_var - 1

                component_time += 1 * product_var

        time_dict[comp_id] = component_time

    return time_dict


def component_transfer_time(
    problem,
    model_graph: nx.DiGraph,
    num_components: int,
    network_graph: nx.DiGraph,
    model_edge_comp_edge_ass_vars: dict[tuple, pulp.LpVariable],
    comp_edge_server_edge_ass_vars: dict[tuple, pulp.LpVariable],
):
    time_dict = {}
    for comp_id in range(num_components):
        component_time = 0
        for server_id in network_graph.nodes:

            for model_edge_id in model_graph.edges:
                for net_edge_id in network_graph.edges:
                    if net_edge_id[0] == server_id:
                        for other_comp_id in range(num_components):

                            model_edge_var = model_edge_comp_edge_ass_vars[
                                (model_edge_id, (comp_id, other_comp_id))
                            ]
                            comp_edge_var = comp_edge_server_edge_ass_vars[
                                ((comp_id, other_comp_id), net_edge_id)
                            ]

                            product_var = pulp.LpVariable(
                                get_next_var_idx(), cat=pulp.LpBinary
                            )

                            problem += product_var <= model_edge_var
                            problem += product_var <= comp_edge_var
                            problem += product_var >= model_edge_var + comp_edge_var - 1

                            component_time += 1 * product_var

        time_dict[comp_id] = component_time

    return time_dict


def component_delay(
    problem,
    num_components: int,
    comp_dependency_vars: dict[tuple, pulp.LpVariable],
    comp_time_dict,
    trans_time_dict,
):
    delay_dict = {}
    for comp_id in range(num_components):
        delay_dict[comp_id] = pulp.LpVariable(get_next_var_idx(), lowBound=0)

    for comp_id in range(num_components):
        comp_delay = delay_dict[comp_id]
        prev_delay_var = pulp.LpVariable(get_next_var_idx(), lowBound=0)

        for other_comp_id in range(num_components):
            if comp_id != other_comp_id and other_comp_id < comp_id:
                problem += prev_delay_var >= delay_dict[other_comp_id]

        problem += (
            comp_delay
            == prev_delay_var + comp_time_dict[comp_id] + trans_time_dict[comp_id]
        )

    return delay_dict


def set_assignment_constraints(
    problem,
    model_graph: nx.DiGraph,
    num_components: int,
    network_graph: nx.DiGraph,
    layer_comp_ass_vars: dict[tuple],
    comp_server_ass_vars: dict[tuple],
    model_edge_comp_edge_ass_vars: dict[tuple],
    comp_edge_server_edge_ass_vars: dict[tuple],
):
    for layer_id in model_graph.nodes:
        sum = 0
        for comp_id in range(num_components):
            sum += layer_comp_ass_vars[(layer_id, comp_id)]
        problem += sum == 1

    for comp_id in range(num_components):
        sum = 0
        for server_id in network_graph.nodes:
            sum += comp_server_ass_vars[(comp_id, server_id)]
        problem += sum == 1

    for mod_edge_id in model_graph.edges:
        sum = 0
        for comp_id in range(num_components):
            for other_comp_id in range(num_components):
                sum += model_edge_comp_edge_ass_vars[
                    (mod_edge_id, (comp_id, other_comp_id))
                ]
        problem += sum == 1

    for comp_id in range(num_components):
        for other_comp_id in range(num_components):
            sum = 0
            for net_edge_id in network_graph.edges:
                sum += comp_edge_server_edge_ass_vars[
                    ((comp_id, other_comp_id), net_edge_id)
                ]
            problem += sum == 1

    pass


def set_flow_constraints(
    problem,
    model_graph: nx.DiGraph,
    num_components: int,
    network_graph: nx.DiGraph,
    layer_comp_ass_vars: dict[tuple],
    comp_server_ass_vars: dict[tuple],
    model_edge_comp_edge_ass_vars: dict[tuple],
    comp_edge_server_edge_ass_vars: dict[tuple],
):
    for mod_edge_id in model_graph.edges:
        first_layer_id = mod_edge_id[0]
        for comp_id in range(num_components):
            first_layer_ass = layer_comp_ass_vars[(first_layer_id, comp_id)]

            sum = 0
            for other_comp_id in range(num_components):
                sum += model_edge_comp_edge_ass_vars[
                    (mod_edge_id, (comp_id, other_comp_id))
                ]

            problem += first_layer_ass == sum

    for mod_edge_id in model_graph.edges:
        second_layer_id = mod_edge_id[1]
        for next_comp_id in range(num_components):
            second_layer_ass = layer_comp_ass_vars[(second_layer_id, next_comp_id)]

            sum = 0
            for comp_id in range(num_components):
                sum += model_edge_comp_edge_ass_vars[
                    (mod_edge_id, (comp_id, next_comp_id))
                ]

            problem += second_layer_ass == sum

    for comp_id, other_comp_id in itertools.product(range(num_components), repeat=2):
        for server_id in network_graph.nodes:
            comp_ass = comp_server_ass_vars[(comp_id, server_id)]

            sum = 0
            for other_server_id in network_graph.nodes:
                sum += comp_edge_server_edge_ass_vars[
                    ((comp_id, other_comp_id), (server_id, other_server_id))
                ]

            problem += comp_ass == sum

    for comp_id, next_comp_id in itertools.product(range(num_components), repeat=2):
        for next_server_id in network_graph.nodes:
            comp_ass = comp_server_ass_vars[(next_comp_id, next_server_id)]

            sum = 0
            for server_id in network_graph.nodes:
                sum += comp_edge_server_edge_ass_vars[
                    ((comp_id, next_comp_id), (server_id, next_server_id))
                ]

            problem += comp_ass == sum


def set_acyclic_constraints(
    problem,
    num_components: int,
    comp_dependency_vars: dict[tuple, pulp.LpVariable],
    model_graph: nx.DiGraph,
    layer_comp_ass_vars: dict[tuple, pulp.LpVariable],
):
    for model_edge_id in model_graph.edges:
        for comp_id, next_comp_id in itertools.product(range(num_components), repeat=2):
            if comp_id != next_comp_id:
                first_ass_var = layer_comp_ass_vars[(model_edge_id[0], comp_id)]
                second_ass_var = layer_comp_ass_vars[(model_edge_id[1], next_comp_id)]

                problem += (
                    first_ass_var + second_ass_var - 1
                    <= comp_dependency_vars[(comp_id, next_comp_id)]
                )

    for comp_id, next_comp_id in itertools.product(range(num_components), repeat=2):
        if comp_id >= next_comp_id:
            dep_var = comp_dependency_vars[(comp_id, next_comp_id)]
            problem += dep_var == 0


def set_input_output_constraints(
    problem,
    model_graph: nx.DiGraph,
    num_components: int,
    layer_comp_ass_vars: dict[tuple, pulp.LpVariable],
    comp_server_ass_vars: dict[tuple, pulp.LpVariable],
    network_graph: nx.DiGraph,
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

    problem += comp_server_ass_vars[(input_comp_id, 0)] == 1
    problem += comp_server_ass_vars[(output_comp_id, 0)] == 1


def main():

    num_components = 3 + 2
    model_graph: nx.DiGraph = build_model_graph()
    network_graph: nx.DiGraph = build_network_graph()

    problem = pulp.LpProblem(name="Assignment", sense=pulp.LpMinimize)

    layer_comp_ass_vars = define_layer_comp_ass_vars(model_graph, num_components)

    comp_server_ass_vars = define_comp_server_ass_vars(num_components, network_graph)

    model_edge_comp_edge_ass_vars = define_model_edge_comp_edge_ass_vars(
        model_graph, num_components
    )

    comp_edge_server_edge_ass_vars = define_model_node_comp_node_ass_vars(
        num_components, network_graph
    )

    comp_dependency_vars = define_comp_dependency_vars(num_components)

    comp_time_dict = component_computation_time(
        problem,
        model_graph,
        num_components,
        network_graph,
        layer_comp_ass_vars,
        comp_server_ass_vars,
    )
    trans_time_dict = component_transfer_time(
        problem,
        model_graph,
        num_components,
        network_graph,
        model_edge_comp_edge_ass_vars,
        comp_edge_server_edge_ass_vars,
    )

    delay_dict = component_delay(
        problem,
        num_components,
        comp_dependency_vars,
        comp_time_dict,
        trans_time_dict,
    )

    set_assignment_constraints(
        problem,
        model_graph,
        num_components,
        network_graph,
        layer_comp_ass_vars,
        comp_server_ass_vars,
        model_edge_comp_edge_ass_vars,
        comp_edge_server_edge_ass_vars,
    )

    set_flow_constraints(
        problem,
        model_graph,
        num_components,
        network_graph,
        layer_comp_ass_vars,
        comp_server_ass_vars,
        model_edge_comp_edge_ass_vars,
        comp_edge_server_edge_ass_vars,
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
        comp_server_ass_vars,
        network_graph,
    )

    out_delay = delay_dict[num_components - 1]

    problem += out_delay

    problem: pulp.LpProblem
    problem.solve(pulp.PULP_CBC_CMD())

    print("Status:", pulp.LpStatus[problem.status])

    quotient_graph = nx.DiGraph()
    for dep_var_key, dep_var in comp_dependency_vars.items():
        if dep_var.value() == 1.0:
            quotient_graph.add_edge(dep_var_key[0], dep_var_key[1])

    print(quotient_graph.edges)


if __name__ == "__main__":
    main()
