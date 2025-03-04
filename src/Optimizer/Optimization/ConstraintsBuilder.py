import pulp
from Graph.Graph import EdgeId, NodeId
from Graph.ModelGraph import ModelGraph, ModelNodeInfo
from Graph.NetworkGraph import NetworkGraph, NetworkNodeInfo
from Optimization.OptimizationKeys import EdgeAssKey, MemoryUseKey, NodeAssKey


class ConstraintsBuilder:

    @staticmethod
    def add_node_assignment_constraints(
        problem: pulp.LpProblem,
        model_graph: ModelGraph,
        node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
        deployment_server_id: NodeId,
    ):
        ## One server per layer!!
        for mod_node_id in model_graph.get_nodes_id():
            var_list = []
            for x_var_key, x_var in node_ass_vars.items():
                if x_var_key.check_model_node_and_name(
                    mod_node_id, model_graph.get_graph_name()
                ):
                    var_list.append(x_var)

            problem += pulp.lpSum(var_list) == 1

        ## Input nodes on server_0
        for inp_node_id in model_graph.get_input_nodes():
            x_var_key = NodeAssKey(
                inp_node_id, deployment_server_id, model_graph.get_graph_name()
            )

            x_var = node_ass_vars[x_var_key]
            problem += x_var == 1

        ## Output node on server_0
        for out_node_id in model_graph.get_output_nodes():
            x_var_key = NodeAssKey(
                out_node_id, deployment_server_id, model_graph.get_graph_name()
            )

            x_var = node_ass_vars[x_var_key]
            problem += x_var == 1
        pass

    def add_edge_assignment_constraints(
        problem: pulp.LpProblem,
        model_graph: ModelGraph,
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    ):
        ## One link per model edge
        for mod_edge_id in model_graph.get_edges_id():
            var_list = []
            for y_var_key, y_var in edge_ass_vars.items():
                if y_var_key.check_model_edge_and_name(
                    mod_edge_id, model_graph.get_graph_name()
                ):
                    var_list.append(y_var)

            problem += pulp.lpSum(var_list) == 1
        pass

    def add_output_flow_constraints(
        problem: pulp.LpProblem,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
        node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    ):
        ## First Flow Balance
        for mod_edge_id in model_graph.get_edges_id():  ## (i, j)
            for src_net_node_id in network_graph.get_nodes_id():  ## Net Node h
                y_sum_vars = []
                for dst_net_node_id in network_graph.get_nodes_id():  ## Net Node k
                    net_edge_id = EdgeId(src_net_node_id, dst_net_node_id)
                    y_var_key = EdgeAssKey(
                        mod_edge_id, net_edge_id, model_graph.get_graph_name()
                    )

                    if edge_ass_vars.get(y_var_key) is not None:
                        y_sum_vars.append(edge_ass_vars[y_var_key])

                x_var_key = NodeAssKey(
                    mod_edge_id.first_node_id,
                    src_net_node_id,
                    model_graph.get_graph_name(),
                )
                x_var = node_ass_vars[x_var_key]

                problem += x_var == pulp.lpSum(y_sum_vars)
        pass

    def add_input_flow_constraints(
        problem: pulp.LpProblem,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
        node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    ):
        ## Second Flow Balance
        for mod_edge_id in model_graph.get_edges_id():  ## (i, j)
            for dst_net_node_id in network_graph.get_nodes_id():  ## Net Node k
                y_sum_vars = []
                for src_net_node_id in network_graph.get_nodes_id():  ## Net Node k
                    net_edge_id = EdgeId(src_net_node_id, dst_net_node_id)
                    y_var_key = EdgeAssKey(
                        mod_edge_id, net_edge_id, model_graph.get_graph_name()
                    )

                    if edge_ass_vars.get(y_var_key) is not None:
                        y_sum_vars.append(edge_ass_vars[y_var_key])

                x_var_key = NodeAssKey(
                    mod_edge_id.second_node_id,
                    dst_net_node_id,
                    model_graph.get_graph_name(),
                )
                x_var = node_ass_vars[x_var_key]

                problem += x_var == pulp.lpSum(y_sum_vars)
        pass

    @staticmethod
    def add_memory_constraints(
        problem: pulp.LpProblem,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
        ass_vars: dict[NodeAssKey, pulp.LpVariable],
        mem_use_vars: dict[MemoryUseKey, pulp.LpVariable],
        requests_num: int,
    ):

        ## Bounding for max definition
        for net_node_id in network_graph.get_nodes_id():
            mem_use_var_key = MemoryUseKey(model_graph.get_graph_name(), net_node_id)
            mem_use_var = mem_use_vars[mem_use_var_key]

            for mod_node_id in model_graph.get_nodes_id():
                x_var_key = NodeAssKey(
                    mod_node_id, net_node_id, model_graph.get_graph_name()
                )
                x_var = ass_vars[x_var_key]

                mod_node_info: ModelNodeInfo = model_graph.get_node_info(mod_node_id)
                mod_node_out_size = mod_node_info.get_node_outputs_size()

                problem += mem_use_var >= x_var * mod_node_out_size

        for net_node_id in network_graph.get_nodes_id():
            sum_vars = []
            for mod_node_id in model_graph.get_nodes_id():
                x_var_key = NodeAssKey(
                    mod_node_id, net_node_id, model_graph.get_graph_name()
                )
                x_var = ass_vars[x_var_key]

                mod_node_info: ModelNodeInfo = model_graph.get_node_info(mod_node_id)
                mod_node_weights_size = mod_node_info.get_node_weights_size()

                sum_vars.append(x_var * mod_node_weights_size)

            mem_use_var_key = MemoryUseKey(model_graph.get_graph_name(), net_node_id)
            mem_use_var = mem_use_vars[mem_use_var_key]

            total_memory = pulp.lpSum(sum_vars) + mem_use_var

            net_node_info: NetworkNodeInfo = network_graph.get_node_info(net_node_id)
            problem += total_memory <= net_node_info.get_available_memory()

    @staticmethod
    def add_energy_constraints():
        ## TODO To Implement
        pass
