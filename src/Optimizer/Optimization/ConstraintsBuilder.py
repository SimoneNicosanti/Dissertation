import networkx as nx
import pulp

from Optimizer.Graph.Graph import NetworkNodeInfo, NodeId
from Optimizer.Optimization import EnergyComputer
from Optimizer.Optimization.OptimizationKeys import EdgeAssKey, MemoryUseKey, NodeAssKey


class ConstraintsBuilder:

    @staticmethod
    def add_node_assignment_constraints(
        problem: pulp.LpProblem,
        model_graph: nx.MultiDiGraph,
        node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
        deployment_server_id: NodeId,
    ):
        ## One server per layer!!
        for mod_node_id in list(model_graph.nodes):
            node_ass_sum = 0
            for x_var_key, x_var in node_ass_vars.items():

                if x_var_key.check_model_node_and_name(
                    mod_node_id, model_graph.graph["name"]
                ):
                    node_ass_sum += x_var

            problem += node_ass_sum == 1

        ## Input nodes on server_0
        generator_nodes = [
            node for node in model_graph.nodes if model_graph.in_degree(node) == 0
        ]
        for inp_node_id in generator_nodes:
            x_var_key = NodeAssKey(
                inp_node_id, deployment_server_id, model_graph.graph["name"]
            )

            x_var = node_ass_vars[x_var_key]
            problem += x_var == 1

        ## Output node on server_0
        receiver_nodes = [
            node for node in model_graph.nodes if model_graph.out_degree(node) == 0
        ]
        for out_node_id in receiver_nodes:
            x_var_key = NodeAssKey(
                out_node_id, deployment_server_id, model_graph.graph["name"]
            )

            x_var = node_ass_vars[x_var_key]
            problem += x_var == 1
        pass

    def add_edge_assignment_constraints(
        problem: pulp.LpProblem,
        model_graph: nx.MultiDiGraph,
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    ):
        ## One link per model edge
        for mod_edge_id in model_graph.edges:
            edge_ass_sum = 0
            for y_var_key, y_var in edge_ass_vars.items():
                if y_var_key.check_model_edge_and_name(
                    mod_edge_id, model_graph.graph["name"]
                ):
                    edge_ass_sum += y_var

            problem += edge_ass_sum == 1
        pass

    def add_output_flow_constraints(
        problem: pulp.LpProblem,
        model_graph: nx.MultiDiGraph,
        network_graph: nx.DiGraph,
        node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    ):
        ## First Flow Balance
        for mod_edge_id in model_graph.edges:  ## (i, j)
            for src_net_node_id in network_graph.nodes:  ## Net Node h
                y_sum_vars = []
                for dst_net_node_id in network_graph.nodes:  ## Net Node k

                    net_edge_id = (src_net_node_id, dst_net_node_id)
                    y_var_key = EdgeAssKey(
                        mod_edge_id, net_edge_id, model_graph.graph["name"]
                    )

                    if edge_ass_vars.get(y_var_key) is not None:
                        y_sum_vars.append(edge_ass_vars[y_var_key])

                x_var_key = NodeAssKey(
                    mod_edge_id[0],
                    src_net_node_id,
                    model_graph.graph["name"],
                )
                x_var = node_ass_vars[x_var_key]

                problem += x_var == pulp.lpSum(y_sum_vars)
        pass

    def add_input_flow_constraints(
        problem: pulp.LpProblem,
        model_graph: nx.MultiDiGraph,
        network_graph: nx.DiGraph,
        node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    ):
        ## Second Flow Balance
        for mod_edge_id in model_graph.edges:  ## (i, j)
            for dst_net_node_id in network_graph.nodes:  ## Net Node k
                y_sum_vars = []
                for src_net_node_id in network_graph.nodes:  ## Net Node k
                    net_edge_id = (src_net_node_id, dst_net_node_id)
                    y_var_key = EdgeAssKey(
                        mod_edge_id, net_edge_id, model_graph.graph["name"]
                    )

                    if edge_ass_vars.get(y_var_key) is not None:
                        y_sum_vars.append(edge_ass_vars[y_var_key])

                x_var_key = NodeAssKey(
                    mod_edge_id[1],
                    dst_net_node_id,
                    model_graph.graph["name"],
                )
                x_var = node_ass_vars[x_var_key]

                problem += x_var == pulp.lpSum(y_sum_vars)
        pass

    @staticmethod
    def add_memory_constraints(
        problem: pulp.LpProblem,
        model_graphs: list[nx.MultiDiGraph],
        network_graph: nx.DiGraph,
        ass_vars: dict[NodeAssKey, pulp.LpVariable],
        mem_use_vars: dict[MemoryUseKey, pulp.LpVariable],
    ):

        ## Bounding for max definition
        for net_node_id in network_graph.nodes:
            for model_graph in model_graphs:
                mem_use_var_key = MemoryUseKey(model_graph.graph["name"], net_node_id)
                mem_use_var = mem_use_vars[mem_use_var_key]

                for mod_node_id in model_graph.nodes:
                    x_var_key = NodeAssKey(
                        mod_node_id, net_node_id, model_graph.graph["name"]
                    )
                    x_var = ass_vars[x_var_key]

                    mod_node_out_size = model_graph.nodes[mod_node_id]["outputs_size"]

                    problem += mem_use_var >= x_var * mod_node_out_size

        for net_node_id in network_graph.nodes:
            used_memory = 0
            for model_graph in model_graphs:
                model_used_memory = 0
                for mod_node_id in model_graph.nodes:
                    x_var_key = NodeAssKey(
                        mod_node_id, net_node_id, model_graph.graph["name"]
                    )
                    x_var = ass_vars[x_var_key]

                    mod_node_weights_size = model_graph.nodes[mod_node_id][
                        "weights_size"
                    ]

                    model_used_memory += x_var * mod_node_weights_size

                mem_use_var_key = MemoryUseKey(model_graph.graph["name"], net_node_id)
                mem_use_var = mem_use_vars[mem_use_var_key]

                used_memory += model_used_memory + mem_use_var

            problem += (
                used_memory
                <= network_graph.nodes[net_node_id][NetworkNodeInfo.AVAILABLE_MEMORY]
            )

    @staticmethod
    def add_energy_constraints(
        problem: pulp.LpProblem,
        model_graphs: list[nx.MultiDiGraph],
        network_graph: nx.DiGraph,
        node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
        requests_num: dict[str, int],
        net_node_id: NodeId,
        device_energy_limit: float,
    ):

        device_energy = EnergyComputer.compute_energy_cost_per_net_node(
            model_graphs,
            network_graph,
            node_ass_vars,
            edge_ass_vars,
            requests_num,
            net_node_id,
        )
        problem += device_energy <= device_energy_limit
        pass
