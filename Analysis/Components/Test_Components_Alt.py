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


def build_network_graph():
    network_graph = nx.DiGraph()
    for i, j in itertools.product(range(2), repeat=2):
        if i == j:
            network_graph.add_edge(i, j, bandwidth=1000)
        else:
            network_graph.add_edge(i, j, bandwidth=7.5)

    for i in range(2):
        if i == 0:
            network_graph.nodes[i]["flops"] = 3 * 10**9
        else:
            network_graph.nodes[i]["flops"] = 5 * 10**9

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


###############################################################################################


def component_computation_time(
    model_graph: nx.DiGraph,
    num_components: int,
    network_graph: nx.DiGraph,
    layer_comp_ass_vars: dict[tuple, pulp.LpVariable],
    comp_server_ass_vars: dict[tuple, pulp.LpVariable],
):
    time_dict = {}

    for comp_id in range(num_components):
        component_time = 0

        for layer_id in model_graph.nodes:
            layer_ass_var = layer_comp_ass_vars[(layer_id, comp_id)]

            for server_id in network_graph.nodes:
                comp_ass_var = comp_server_ass_vars[(comp_id, server_id)]

                component_time += (
                    (
                        model_graph.nodes[layer_id]["flops"]
                        / network_graph.nodes[server_id]["flops"]
                    )
                    * comp_ass_var
                    * layer_ass_var
                )

        time_dict[comp_id] = component_time

    return time_dict


def component_transfer_time(
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

                            if comp_id != other_comp_id:
                                component_time += (
                                    (
                                        model_graph.edges[model_edge_id][
                                            "tot_tensor_size"
                                        ]
                                        / network_graph.edges[net_edge_id]["bandwidth"]
                                    )
                                    * model_edge_var
                                    * comp_edge_var
                                )

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


def set_component_assignment_constraints(
    problem,
    num_components: int,
    network_graph: nx.DiGraph,
    comp_server_ass_vars: dict[tuple, pulp.LpVariable],
    comp_edge_server_edge_ass_vars: dict[tuple, pulp.LpVariable],
):

    ## Each component on one server
    for comp_id in range(num_components):
        sum = 0
        for server_id in network_graph.nodes:
            sum += comp_server_ass_vars[(comp_id, server_id)]
        problem += sum == 1

    ## Each network edge on one component edge
    for comp_id in range(num_components):
        for other_comp_id in range(num_components):
            sum = 0
            for net_edge_id in network_graph.edges:
                sum += comp_edge_server_edge_ass_vars[
                    ((comp_id, other_comp_id), net_edge_id)
                ]
            problem += sum == 1

    ## For sending flow
    for comp_id, other_comp_id in itertools.product(range(num_components), repeat=2):
        for server_id in network_graph.nodes:
            comp_ass = comp_server_ass_vars[(comp_id, server_id)]

            sum = 0
            for other_server_id in network_graph.nodes:
                sum += comp_edge_server_edge_ass_vars[
                    ((comp_id, other_comp_id), (server_id, other_server_id))
                ]

            problem += comp_ass == sum

    ## For receiving flow
    for comp_id, next_comp_id in itertools.product(range(num_components), repeat=2):
        for next_server_id in network_graph.nodes:
            comp_ass = comp_server_ass_vars[(next_comp_id, next_server_id)]

            sum = 0
            for server_id in network_graph.nodes:
                sum += comp_edge_server_edge_ass_vars[
                    ((comp_id, next_comp_id), (server_id, next_server_id))
                ]

            problem += comp_ass == sum

    input_comp_id = 0
    output_comp_id = num_components - 1
    ## Constraint on components position
    problem += comp_server_ass_vars[(input_comp_id, 0)] == 1
    problem += comp_server_ass_vars[(output_comp_id, 0)] == 1
    pass


def set_layer_assignment_constraints(
    problem,
    model_graph: nx.DiGraph,
    num_components: int,
    layer_comp_ass_vars: dict[tuple, pulp.LpVariable],
    model_edge_comp_edge_ass_vars: dict[tuple, pulp.LpVariable],
    comp_dependency_vars: dict[tuple, pulp.LpVariable],
):

    ## Each layer on one component
    for layer_id in model_graph.nodes:
        sum = 0
        for comp_id in range(num_components):
            sum += layer_comp_ass_vars[(layer_id, comp_id)]
        problem += sum == 1

    ## Each model edge on one component edge
    for mod_edge_id in model_graph.edges:
        sum = 0
        for comp_id in range(num_components):
            for other_comp_id in range(num_components):
                sum += model_edge_comp_edge_ass_vars[
                    (mod_edge_id, (comp_id, other_comp_id))
                ]
        problem += sum == 1

    ## Respect of Sending Flow
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

    ## Respect of receiving Flow
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

    # ## Constraint on dependency of components
    # for model_edge_id in model_graph.edges:
    #     for comp_id, next_comp_id in itertools.product(range(num_components), repeat=2):
    #         if comp_id != next_comp_id:
    #             first_ass_var = layer_comp_ass_vars[(model_edge_id[0], comp_id)]
    #             second_ass_var = layer_comp_ass_vars[(model_edge_id[1], next_comp_id)]

    #             problem += (
    #                 first_ass_var + second_ass_var - 1
    #                 <= comp_dependency_vars[(comp_id, next_comp_id)]
    #             )

    for comp_id, next_comp_id in itertools.product(range(num_components), repeat=2):
        comp_edge_id = (comp_id, next_comp_id)
        if comp_id != next_comp_id:
            dep_var = comp_dependency_vars[comp_edge_id]
            sum = 0
            tot_elems = 0
            for model_edge_id in model_graph.edges:
                sum += model_edge_comp_edge_ass_vars[(model_edge_id, comp_edge_id)]
                tot_elems += 1

            problem += dep_var >= sum / tot_elems

    ## Dependency Matrix is Triangular
    for comp_id, next_comp_id in itertools.product(range(num_components), repeat=2):
        if comp_id >= next_comp_id:
            dep_var = comp_dependency_vars[(comp_id, next_comp_id)]
            problem += dep_var == 0

    input_comp_id = 0
    output_comp_id = num_components - 1

    ## Placing generator node on input component
    for layer_id in model_graph.nodes:
        if model_graph.nodes[layer_id].get("generator", False):
            layer_ass = layer_comp_ass_vars[(layer_id, input_comp_id)]
            problem += layer_ass == 1

    ## Placing receiver node on output component
    for layer_id in model_graph.nodes:
        if model_graph.nodes[layer_id].get("receiver", False):
            layer_ass = layer_comp_ass_vars[(layer_id, output_comp_id)]
            problem += layer_ass == 1

    ## Placing one node in input component
    input_comp_tot_nodes = 0
    for layer_id in model_graph.nodes:
        ass_var = layer_comp_ass_vars[(layer_id, input_comp_id)]
        input_comp_tot_nodes += ass_var
    problem += input_comp_tot_nodes == 1

    ## Placing one node in output component
    output_comp_tot_nodes = 0
    for layer_id in model_graph.nodes:
        ass_var = layer_comp_ass_vars[(layer_id, output_comp_id)]
        output_comp_tot_nodes += ass_var
    problem += output_comp_tot_nodes == 1


def init_problem(
    num_components: int,
    network_graph: nx.DiGraph,
    comp_ass_values: dict,
    comp_edge_ass_values: dict,
):
    for component_id in range(num_components):
        for server_id in network_graph.nodes:
            comp_ass_values[(component_id, server_id)] = 1 if server_id == 0 else 0

    for comp_id, next_comp_id in itertools.product(range(num_components), repeat=2):
        comp_edge_id = (comp_id, next_comp_id)
        for net_edge_id in network_graph.edges:
            if net_edge_id[0] == 0 and net_edge_id[1] == 0:
                comp_edge_ass_values[(comp_edge_id, net_edge_id)] = 1
            else:
                comp_edge_ass_values[(comp_edge_id, net_edge_id)] = 0


def optimize_layer_assignments(
    problem,
    model_graph: nx.DiGraph,
    num_components: int,
    network_graph: nx.DiGraph,
    layer_comp_ass_vars: dict,
    comp_server_ass_values: dict,
    model_edge_comp_edge_ass_vars: dict,
    comp_edge_server_edge_ass_values: dict,
    comp_dependency_vars: dict,
):

    set_layer_assignment_constraints(
        problem,
        model_graph,
        num_components,
        layer_comp_ass_vars,
        model_edge_comp_edge_ass_vars,
        comp_dependency_vars,
    )

    component_comp_time = component_computation_time(
        model_graph,
        num_components,
        network_graph,
        layer_comp_ass_vars,
        comp_server_ass_values,
    )

    component_transf_time = component_transfer_time(
        model_graph,
        num_components,
        network_graph,
        model_edge_comp_edge_ass_vars,
        comp_edge_server_edge_ass_values,
    )

    delay_dict = component_delay(
        problem,
        num_components,
        comp_dependency_vars,
        component_comp_time,
        component_transf_time,
    )

    out_delay = delay_dict[num_components - 1]

    problem += out_delay

    problem: pulp.LpProblem
    problem.solve(
        pulp.SCIP_PY(timeLimit=90)
        # pulp.PULP_CBC_CMD(
        #     msg=False, timeLimit=120, presolve=True, cuts=True, strong=True
        # )
    )

    print("Status:", pulp.LpStatus[problem.status])
    print("Delay:", out_delay.varValue)

    return out_delay.varValue
    pass


def optimize_component_assignments(
    problem,
    model_graph: nx.DiGraph,
    num_components: int,
    network_graph: nx.DiGraph,
    layer_comp_ass_values: dict,
    comp_server_ass_vars: dict,
    mod_edge_comp_edge_ass_values: dict,
    comp_edge_server_edge_ass_vars: dict,
    comp_dependency_values: dict,
):

    set_component_assignment_constraints(
        problem,
        num_components,
        network_graph,
        comp_server_ass_vars,
        comp_edge_server_edge_ass_vars,
    )

    component_comp_time = component_computation_time(
        model_graph,
        num_components,
        network_graph,
        layer_comp_ass_values,
        comp_server_ass_vars,
    )

    component_transf_time = component_transfer_time(
        model_graph,
        num_components,
        network_graph,
        mod_edge_comp_edge_ass_values,
        comp_edge_server_edge_ass_vars,
    )

    delay_dict = component_delay(
        problem,
        num_components,
        comp_dependency_values,
        component_comp_time,
        component_transf_time,
    )

    out_delay = delay_dict[num_components - 1]

    problem += out_delay

    problem: pulp.LpProblem
    problem.solve(
        pulp.SCIP_PY(timeLimit=90)
        # pulp.PULP_CBC_CMD(
        #     msg=False, timeLimit=120, presolve=True, cuts=True, strong=True
        # )
    )

    print("Status:", pulp.LpStatus[problem.status])
    print("Delay:", out_delay.varValue)

    return out_delay.varValue


def main():

    num_components = 10 + 2
    model_graph: nx.DiGraph = build_model_graph()
    network_graph: nx.DiGraph = build_network_graph()

    layer_comp_ass_vars = define_layer_comp_ass_vars(model_graph, num_components)

    comp_server_ass_vars = define_comp_server_ass_vars(num_components, network_graph)

    model_edge_comp_edge_ass_vars = define_model_edge_comp_edge_ass_vars(
        model_graph, num_components
    )

    comp_edge_server_edge_ass_vars = define_model_node_comp_node_ass_vars(
        num_components, network_graph
    )

    comp_dependency_vars = define_comp_dependency_vars(num_components)

    layer_comp_ass_values = {}
    comp_server_ass_values = {}
    mod_edge_comp_edge_ass_values = {}
    comp_edge_server_edge_ass_values = {}
    comp_dependency_values = {}

    init_problem(
        num_components,
        network_graph,
        comp_server_ass_values,
        comp_edge_server_edge_ass_values,
    )

    optimized_values = []
    prev_opt_val = +float("inf")
    for i in range(0, 10):
        print(f"Running Iteration >>> {i}")
        problem = pulp.LpProblem(name="Assignment", sense=pulp.LpMinimize)

        if i % 2 == 0:
            opt_value = optimize_layer_assignments(
                problem,
                model_graph,
                num_components,
                network_graph,
                layer_comp_ass_vars,
                comp_server_ass_values,
                model_edge_comp_edge_ass_vars,
                comp_edge_server_edge_ass_values,
                comp_dependency_vars,
            )

            if opt_value <= prev_opt_val:
                for key in layer_comp_ass_vars:
                    layer_comp_ass_values[key] = layer_comp_ass_vars[key].varValue

                for key in model_edge_comp_edge_ass_vars:
                    mod_edge_comp_edge_ass_values[key] = model_edge_comp_edge_ass_vars[
                        key
                    ].varValue

                for key in comp_dependency_vars:
                    comp_dependency_values[key] = comp_dependency_vars[key].varValue

        else:
            opt_value = optimize_component_assignments(
                problem,
                model_graph,
                num_components,
                network_graph,
                layer_comp_ass_values,
                comp_server_ass_vars,
                mod_edge_comp_edge_ass_values,
                comp_edge_server_edge_ass_vars,
                comp_dependency_values,
            )

            if opt_value <= prev_opt_val:
                for key in comp_server_ass_vars:
                    comp_server_ass_values[key] = comp_server_ass_vars[key].varValue

                for key in comp_edge_server_edge_ass_vars:
                    comp_edge_server_edge_ass_values[key] = (
                        comp_edge_server_edge_ass_vars[key].varValue
                    )

        if opt_value <= prev_opt_val:
            optimized_values.append(opt_value)
            prev_opt_val = opt_value
        else:
            break

    valid_components = set()
    for key in layer_comp_ass_values:
        if layer_comp_ass_values[key] == 1:
            valid_components.add(key[1])

    dependency_graph = nx.DiGraph()
    for key in comp_dependency_values:
        if (
            key[0] != key[1]
            and comp_dependency_values[key] == 1
            and key[0] in valid_components
            and key[1] in valid_components
        ):
            dependency_graph.add_edge(key[0], key[1])

    for key in comp_server_ass_values:
        if comp_server_ass_values[key] == 1:

            print(key)

    print(optimized_values)
    pos = graphviz_layout(dependency_graph, prog="dot")
    nx.draw(dependency_graph, pos=pos, with_labels=True)
    plt.savefig("dependency_graph.png")


if __name__ == "__main__":
    main()
