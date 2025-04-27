import json

import networkx as nx
import numpy as np
import pygad


def build_model_graph():
    with open("yolo11n-seg.json") as f:
        data = json.load(f)

    # Converte il JSON in grafo NetworkX
    model_graph = nx.readwrite.node_link_graph(data)
    return model_graph


model_graph: nx.DiGraph = build_model_graph()
num_components = 2 + 5
num_servers = 2


def acyclicity_constraint(solution):
    global model_graph, num_components, num_servers
    layer_assignments = solution[: len(model_graph.nodes)]

    part_graph = nx.DiGraph()
    for edge_id in model_graph.edges:
        first_node_idx = model_graph.nodes[edge_id[0]]["idx"]
        second_node_idx = model_graph.nodes[edge_id[1]]["idx"]

        first_layer_ass = layer_assignments[first_node_idx]
        second_layer_ass = layer_assignments[second_node_idx]

        if (
            first_layer_ass != second_layer_ass
            and (first_layer_ass, second_layer_ass) not in part_graph.edges
        ):
            part_graph.add_edge(first_layer_ass, second_layer_ass, count=0)

        if first_layer_ass > second_layer_ass:
            part_graph[first_layer_ass][second_layer_ass]["count"] += 1

    if nx.is_directed_acyclic_graph(part_graph):
        tot = 0
    else:
        tot = sum(part_graph[edge[0]][edge[1]]["count"] for edge in part_graph.edges)

    return -tot


def computational_fitness_value(solution):
    global model_graph, num_components, num_servers

    layer_assignments = solution[: len(model_graph.nodes)]
    comp_assignments = solution[len(model_graph.nodes) :]

    tot = 0
    for layer_id in model_graph.nodes:
        layer_idx = model_graph.nodes[layer_id]["idx"]
        layer_ass = layer_assignments[layer_idx]

        comp_ass = comp_assignments[layer_ass]

        if comp_ass == 0:
            tot += model_graph.nodes[layer_id]["flops"] / (2.5 * 1e9)
        else:
            tot += model_graph.nodes[layer_id]["flops"] / (5 * 1e9)

    return tot


def transfer_fitness_value(solution):
    global model_graph, num_components, num_servers

    layer_assignments = solution[: len(model_graph.nodes)]
    comp_assignments = solution[len(model_graph.nodes) :]

    tot = 0
    for edge_id in model_graph.edges:
        first_node_idx = model_graph.nodes[edge_id[0]]["idx"]
        second_node_idx = model_graph.nodes[edge_id[1]]["idx"]

        first_layer_ass = layer_assignments[first_node_idx]
        second_layer_ass = layer_assignments[second_node_idx]

        if first_layer_ass == second_layer_ass:
            tot += 0
        else:

            first_comp_ass = comp_assignments[first_layer_ass]
            second_comp_ass = comp_assignments[second_layer_ass]

            if first_comp_ass == second_comp_ass:
                tot += 0
            else:
                tot += model_graph.edges[edge_id]["tot_tensor_size"] / 2

    return tot


def check_valid_assignmets(solution):
    global model_graph, num_components, num_servers

    layer_assignments = solution[: len(model_graph.nodes)]
    comp_assignments = solution[len(model_graph.nodes) :]

    tot = 0
    for layer_id in model_graph.nodes:
        layer_idx = model_graph.nodes[layer_id]["idx"]
        layer_ass = layer_assignments[layer_idx]

        if model_graph.nodes[layer_id].get("generator", False):
            if layer_ass != 0:
                tot += 1

        if model_graph.nodes[layer_id].get("receiver", False):
            if layer_ass != num_components - 1:
                tot += 1

    if comp_assignments[0] != 0:
        tot += 1

    if comp_assignments[num_components - 1] != 0:
        tot += 1

    return -tot


def initial_population(
    model_graph: nx.DiGraph, num_components: int, num_servers: int, pop_size: int
):
    gene_space = []
    for comp_idx in range(num_components):
        layer_ass = [comp_idx] * len(model_graph.nodes)
        layer_ass[0] = 0
        layer_ass[-1] = num_components - 1
        for server_idx in range(num_servers):
            comp_ass = [server_idx] * num_components
            comp_ass[0] = 0
            comp_ass[-1] = 0

            gene = layer_ass + comp_ass
            gene_space.append(gene)

    return gene_space

    # for _ in range(pop_size):
    #     random_layers_ass = np.random.randint(0, num_components, len(model_graph.nodes))
    #     random_layers_ass[0] = 0
    #     random_layers_ass[-1] = num_components - 1

    #     random_comp_ass = np.random.randint(0, num_servers, num_components)
    #     random_comp_ass[0] = 0
    #     random_comp_ass[-1] = 0

    #     random_line = np.concatenate((random_layers_ass, random_comp_ass))
    #     random_list.append(random_line)

    # return random_list


def fitness_func(instance, solution, solution_idx):

    return (
        transfer_fitness_value(solution)
        + computational_fitness_value(solution)
        + acyclicity_constraint(solution)
        + check_valid_assignmets(solution)
    )
    pass


def print_progress(ga_instance):
    print(
        f"Generazione {ga_instance.generations_completed} - Fitness migliore: {ga_instance.best_solution()[1]}"
    )


def main():

    ## Fist part for layer mapping
    ## Second part for component mapping
    global num_components, num_servers, model_graph

    gene_space = []
    for _ in model_graph.nodes:
        gene_space.append(range(num_components))

    for _ in range(num_components):
        gene_space.append(range(num_servers))

    gen_alg = pygad.GA(
        num_generations=1_000,
        num_parents_mating=10,
        fitness_func=fitness_func,
        gene_type=int,
        initial_population=initial_population(
            model_graph, num_components, num_servers, 50
        ),
        on_generation=print_progress,
        mutation_probability=0.8,
        crossover_probability=0.8,
        gene_space=gene_space,
        parallel_processing=("process", None),
    )

    gen_alg.run()

    gen_alg.plot_fitness()
    pass


if __name__ == "__main__":
    main()
