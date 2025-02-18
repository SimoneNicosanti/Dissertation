import pulp
from Graph.Model.ModelGraph import ModelGraph
from Graph.Network.NetworkGraph import NetworkGraph
from GraphId import EdgeId, NodeId


class OptimizerClass:

    def __init__(self):
        pass

    def optimize(self, model_graph: ModelGraph, network_graph: NetworkGraph) -> None:
        problem: pulp.LpProblem = pulp.LpProblem("Partitioning", pulp.LpMinimize)

        assignment_vars: dict[tuple, pulp.LpVariable] = self.add_assignemt_vars(
            problem, model_graph, network_graph
        )
        edge_vars: dict[tuple, pulp.LpVariable] = self.add_edge_vars(
            problem, model_graph, network_graph
        )

        self.add_boundaries(
            problem, model_graph, network_graph, assignment_vars, edge_vars
        )

        ## TODO Add quantization penalty

        compute_latency = self.compute_latency(
            model_graph, network_graph, assignment_vars
        )
        transmission_latency = self.transmission_latency(
            model_graph, network_graph, edge_vars
        )

        compute_energy = self.compute_energy(
            model_graph, network_graph, assignment_vars
        )
        # transmission_energy = self.transmission_energy()

        # problem += (compute_latency + transmission_latency) + (
        #     compute_energy + transmission_energy
        # )

        problem += compute_latency + transmission_latency + compute_energy

        problem.solve(pulp.GLPK_CMD())

        for var in problem.variables():
            print(f"{var.name} = {var.varValue}")

        # Print the objective function value
        print(f"Objective value = {pulp.value(problem.objective)}")

        problem.writeLP("solved_problem.lp")
        pass

    def add_boundaries(
        self,
        problem: pulp.LpProblem,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
        assignment_vars: dict[tuple, pulp.LpVariable],
        edge_vars: dict[tuple, pulp.LpVariable],
    ):
        ## One server per layer!!
        for node in model_graph.get_nodes():
            var_list = []
            for x_var_key, x_var in assignment_vars.items():
                if x_var_key[0] == node.get_node_id():
                    var_list.append(x_var)
            sum = pulp.lpSum(var_list)

            problem += sum == 1

        ## First Flow Balance
        for model_edge in model_graph.get_edges():  ## (i, j)
            for src_net_node in network_graph.get_nodes():  ## Net Node h
                y_sum_vars = []
                for dst_net_node in network_graph.get_nodes():  ## Net Node k
                    model_edge_id = model_edge.get_edge_id()
                    network_edge_id = EdgeId(
                        src_net_node.get_node_id(), dst_net_node.get_node_id()
                    )
                    y_var_key = (model_edge_id, network_edge_id)

                    if edge_vars.get(y_var_key) is not None:
                        y_sum_vars.append(edge_vars[y_var_key])

                x_var_key = (
                    model_edge.get_edge_id().get_first_node_id(),
                    src_net_node.get_node_id(),
                )
                x_var = assignment_vars[x_var_key]

                problem += x_var == pulp.lpSum(y_sum_vars)

        ## Second Flow Balance
        for model_edge in model_graph.get_edges():  ## (i, j)
            for dst_net_node in network_graph.get_nodes():  ## Net Node k
                y_sum_vars = []
                for src_net_node in network_graph.get_nodes():  ## Net Node k
                    model_edge_id = model_edge.get_edge_id()
                    network_edge_id = EdgeId(
                        src_net_node.get_node_id(), dst_net_node.get_node_id()
                    )
                    y_var_key = (model_edge_id, network_edge_id)

                    if edge_vars.get(y_var_key) is not None:
                        y_sum_vars.append(edge_vars[y_var_key])

                x_var_key = (
                    model_edge.get_edge_id().get_second_node_id(),
                    dst_net_node.get_node_id(),
                )
                x_var = assignment_vars[x_var_key]

                problem += x_var == pulp.lpSum(y_sum_vars)

    def compute_latency(
        self,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
        assignment_vars: dict[tuple, pulp.LpVariable],
    ):

        sum = 0
        for mod_node in model_graph.get_nodes():
            for net_node in network_graph.get_nodes():
                x_var_key = (mod_node.get_node_id(), net_node.get_node_id())
                x_var = assignment_vars[x_var_key]
                sum += x_var * (
                    mod_node.get_node_info().get_info("node_flops")
                    / net_node.get_node_info().get_info("flops_per_sec")
                )

        return sum

    def transmission_latency(
        self,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
        edge_vars: dict[tuple, pulp.LpVariable],
    ):
        sum = 0
        for model_edge in model_graph.get_edges():
            for network_edge in network_graph.get_edges():
                y_var_key = (model_edge.get_edge_id(), network_edge.get_edge_id())
                y_var = edge_vars[y_var_key]
                sum += y_var * (
                    model_edge.get_edge_info().get_info("data_size")
                    / network_edge.get_edge_info().get_info("bandwidth")
                )
        return sum

    def compute_energy(
        self,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
        assignment_vars: dict[tuple, pulp.LpVariable],
    ):

        sum = 0
        for net_node in network_graph.get_nodes():
            net_node_sum = 0
            for mod_node in model_graph.get_nodes():
                x_var_key = (mod_node.get_node_id(), net_node.get_node_id())
                x_var = assignment_vars[x_var_key]

                net_node_sum += x_var * (
                    mod_node.get_node_info().get_info("node_flops")
                    / net_node.get_node_info().get_info("flops_per_sec")
                )

            sum += net_node_sum * net_node.get_node_info().get_info(
                "comp_energy_per_sec"
            )

        return sum

    def transmission_energy(self):
        pass

    def add_assignemt_vars(
        self,
        problem: pulp.LpProblem,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
    ) -> dict[tuple, pulp.LpVariable]:
        vars_table: dict[tuple[NodeId, NodeId], pulp.LpVariable] = {}
        for modelNode in model_graph.get_nodes():
            for networkNode in network_graph.get_nodes():
                ## TODO >> Add quantization!!

                var_name: str = self.__build_assignment_var_name(
                    modelNode.get_node_id(), networkNode.get_node_id()
                )
                lp_variable = pulp.LpVariable(var_name, cat="Binary")

                problem.addVariable(lp_variable)

                table_key = (modelNode.get_node_id(), networkNode.get_node_id())
                vars_table[table_key] = lp_variable

        return vars_table

    def add_edge_vars(
        self,
        problem: pulp.LpProblem,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
    ):

        ## TODO Add quantization
        vars_table: dict[tuple[EdgeId, EdgeId], pulp.LpVariable] = {}
        for modelEdge in model_graph.get_edges():
            for networkEdge in network_graph.get_edges():

                var_name: str = self.__build_edge_var_name(
                    modelEdge.get_edge_id(), networkEdge.get_edge_id()
                )
                lp_variable = pulp.LpVariable(var_name, cat="Binary")

                problem.addVariable(lp_variable)

                table_key = (modelEdge.get_edge_id(), networkEdge.get_edge_id())
                vars_table[table_key] = lp_variable

        return vars_table

    def __build_assignment_var_name(self, modelNode: NodeId, networkNode: NodeId):
        return "x_({})({})".format(
            modelNode.get_node_id_str(), networkNode.get_node_id_str()
        )

    def __build_edge_var_name(self, modelEdge: EdgeId, networkEdge: EdgeId):
        return "y_({})({})".format(
            modelEdge.get_edge_id_str(), networkEdge.get_edge_id_str()
        )
