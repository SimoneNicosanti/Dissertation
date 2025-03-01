from dataclasses import dataclass

import pulp
from Graph.Graph import EdgeId, GraphInfo, NodeId
from Graph.ModelGraph import ModelEdgeInfo, ModelGraph, ModelNodeInfo
from Graph.NetworkGraph import NetworkEdgeInfo, NetworkGraph, NetworkNodeInfo
from Optimization.SolvedProblemInfo import SolvedProblemInfo


@dataclass
class OptimizationParams:
    compute_latency_weight: float
    transmission_latency_weight: float
    compute_energy_weight: float
    transmission_energy_weight: float

    device_max_energy: float

    requests_number: dict[str, int]


@dataclass(frozen=True)
class NodeAssKey:
    mod_node_id: NodeId
    net_node_id: NodeId


@dataclass(frozen=True)
class EdgeAssKey:
    mod_edge_id: EdgeId
    net_edge_id: EdgeId


class OptimizationHandler:

    def __init__(self):
        pass

    def optimize(
        self,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
        deployment_server: NodeId,
        optimization_params: OptimizationParams = None,
    ) -> SolvedProblemInfo:
        problem: pulp.LpProblem = pulp.LpProblem("Partitioning", pulp.LpMinimize)

        node_ass_vars: dict[NodeAssKey, pulp.LpVariable] = (
            self.define_node_assignment_vars(problem, model_graph, network_graph)
        )
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable] = (
            self.define_edge_assignment_vars(problem, model_graph, network_graph)
        )

        self.add_constraints(
            problem,
            model_graph,
            network_graph,
            deployment_server,
            node_ass_vars,
            edge_ass_vars,
        )

        compute_latency = self.computation_latency(
            model_graph, network_graph, node_ass_vars
        )

        transmission_latency = self.transmission_latency(
            model_graph, network_graph, edge_ass_vars
        )

        # compute_energy = self.computation_energy(
        #     model_graph, network_graph, node_ass_vars
        # )
        # transmission_energy = self.transmission_energy(
        #     model_graph, network_graph, edge_ass_vars
        # )

        ## TODO Add quantization
        problem += (
            compute_latency
            + transmission_latency
            # + compute_energy
            # + transmission_energy
        )

        problem.solve(pulp.GLPK_CMD())

        with open("VarFile.txt", "w") as f:
            for var in problem.variables():
                f.write(f"{var.name} = {var.varValue}\n")

        # # Print the objective function value
        # print(f"Objective value = {pulp.value(problem.objective)}")

        problem.writeLP("solved_problem.lp")

        solved_problem_info: SolvedProblemInfo = self.build_solved_problem_info(
            problem, node_ass_vars, edge_ass_vars
        )

        return solved_problem_info

    def build_solved_problem_info(
        self,
        problem: pulp.LpProblem,
        node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    ) -> SolvedProblemInfo:

        if pulp.LpStatus[problem.status] != pulp.LpStatus[pulp.LpStatusOptimal]:
            ## Problem could not be solved
            return SolvedProblemInfo(problem_solved=False, solution_value=float("inf"))

        solved_problem_info = SolvedProblemInfo(
            problem_solved=True, solution_value=problem.objective.value()
        )
        for var in problem.variables():
            if var.name.startswith("x_"):
                ## Handle node assignment var
                for node_ass_key, node_ass_var in node_ass_vars.items():
                    if var.name == node_ass_var.name and var.varValue == 1.0:
                        mod_node_id = node_ass_key.mod_node_id
                        net_node_id = node_ass_key.net_node_id
                        solved_problem_info.put_node_assignment(
                            net_node_id, mod_node_id
                        )
                        break
                pass
            elif var.name.startswith("y_"):
                ## Handle edge assignment var
                for edge_ass_key, edge_ass_var in edge_ass_vars.items():
                    if var.name == edge_ass_var.name and var.varValue == 1.0:
                        mod_edge_id = edge_ass_key.mod_edge_id
                        net_edge_id = edge_ass_key.net_edge_id
                        solved_problem_info.put_edge_assignment(
                            net_edge_id, mod_edge_id
                        )
                        break
            else:
                ## There should be no other variables
                pass

        return solved_problem_info

    def add_constraints(
        self,
        problem: pulp.LpProblem,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
        deployment_server_id: NodeId,
        node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    ):
        ## One server per layer!!
        for mod_node_id in model_graph.get_nodes_id():
            var_list = []
            for x_var_key, x_var in node_ass_vars.items():
                if x_var_key.mod_node_id == mod_node_id:
                    var_list.append(x_var)

            problem += pulp.lpSum(var_list) == 1

        ## One link per model edge
        for mod_edge_id in model_graph.get_edges_id():
            var_list = []
            for y_var_key, y_var in edge_ass_vars.items():
                if y_var_key.mod_edge_id == mod_edge_id:
                    var_list.append(y_var)

            problem += pulp.lpSum(var_list) == 1

        ## First Flow Balance
        for mod_edge_id in model_graph.get_edges_id():  ## (i, j)
            for src_net_node_id in network_graph.get_nodes_id():  ## Net Node h
                y_sum_vars = []
                for dst_net_node_id in network_graph.get_nodes_id():  ## Net Node k
                    net_edge_id = EdgeId(src_net_node_id, dst_net_node_id)
                    y_var_key = EdgeAssKey(mod_edge_id, net_edge_id)

                    if edge_ass_vars.get(y_var_key) is not None:
                        y_sum_vars.append(edge_ass_vars[y_var_key])

                x_var_key = NodeAssKey(mod_edge_id.first_node_id, src_net_node_id)
                x_var = node_ass_vars[x_var_key]

                problem += x_var == pulp.lpSum(y_sum_vars)

        ## Second Flow Balance
        for mod_edge_id in model_graph.get_edges_id():  ## (i, j)
            for dst_net_node_id in network_graph.get_nodes_id():  ## Net Node k
                y_sum_vars = []
                for src_net_node_id in network_graph.get_nodes_id():  ## Net Node k
                    net_edge_id = EdgeId(src_net_node_id, dst_net_node_id)
                    y_var_key = EdgeAssKey(mod_edge_id, net_edge_id)

                    if edge_ass_vars.get(y_var_key) is not None:
                        y_sum_vars.append(edge_ass_vars[y_var_key])

                x_var_key = NodeAssKey(
                    mod_edge_id.second_node_id,
                    dst_net_node_id,
                )
                x_var = node_ass_vars[x_var_key]

                problem += x_var == pulp.lpSum(y_sum_vars)

        ## Input nodes on server_0
        for inp_node_id in model_graph.get_input_nodes():
            x_var_key = NodeAssKey(inp_node_id, deployment_server_id)

            x_var = node_ass_vars[x_var_key]
            problem += x_var == 1

        for out_node_id in model_graph.get_output_nodes():
            x_var_key = NodeAssKey(out_node_id, deployment_server_id)

            x_var = node_ass_vars[x_var_key]
            problem += x_var == 1

        # x_var_key = NodeAssKey(NodeId("Conv__435"), deployment_server_id)
        # x_var = node_ass_vars[x_var_key]
        # problem += x_var == 1

    def computation_latency(
        self,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
        node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    ):

        sum_elems = []
        max_comp_latency = 0
        for mod_node_id in model_graph.get_nodes_id():
            for net_node_id in network_graph.get_nodes_id():
                x_var_key = NodeAssKey(mod_node_id, net_node_id)
                x_var = node_ass_vars[x_var_key]

                comp_time = self.__get_computation_time(
                    model_graph.get_node_info(mod_node_id),
                    network_graph.get_node_info(net_node_id),
                )
                max_comp_latency = max(max_comp_latency, comp_time)
                sum_elems.append(x_var * comp_time)

        return pulp.lpSum(sum_elems) / max_comp_latency

    def transmission_latency(
        self,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    ):
        sum_elems = []
        max_trans_latency = 0
        for net_node_id in network_graph.get_nodes_id():
            for mod_edge_id in model_graph.get_edges_id():
                for net_edge_id in network_graph.get_edges_id():
                    if net_edge_id.first_node_id == net_node_id:
                        y_var_key = EdgeAssKey(
                            mod_edge_id,
                            net_edge_id,
                        )
                        y_var = edge_ass_vars[y_var_key]

                        trans_time = self.__get_transmission_time(
                            model_graph.get_edge_info(mod_edge_id),
                            network_graph.get_edge_info(net_edge_id),
                            net_edge_id,
                        )

                        sum_elems.append(y_var * trans_time)

                        max_trans_latency = max(
                            max_trans_latency,
                            trans_time,
                        )

        return pulp.lpSum(sum_elems) / max_trans_latency

    def computation_energy(
        self,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
        node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
    ):
        ## TODO Scale Min-Max

        tot_sum_elems = []
        for net_node_id in network_graph.get_nodes_id():
            net_node_sum_elems = []
            for mod_node_id in model_graph.get_nodes_id():
                x_var_key = NodeAssKey(mod_node_id, net_node_id)
                x_var = node_ass_vars[x_var_key]

                net_node_sum_elems.append(
                    x_var
                    * self.__get_computation_time(
                        model_graph.get_node_info(mod_node_id),
                        network_graph.get_node_info(net_node_id),
                    )
                )

            net_node_sum = pulp.lpSum(net_node_sum_elems)
            tot_sum_elems.append(
                net_node_sum
                * network_graph.get_node_info(net_node_id).get_info(
                    GraphInfo.NET_NODE_COMP_ENERGY_PER_SEC
                )
            )

        return pulp.lpSum(tot_sum_elems)

    def transmission_energy(
        self,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    ):
        ## TODO Scale Min-Max

        tot_sum_elems = []
        for net_node_id in network_graph.get_nodes_id():
            net_node_sum_elems = []
            for mod_edge_id in model_graph.get_edges_id():
                for dst_net_node_id in network_graph.get_nodes_id():
                    net_edge_id = EdgeId(net_node_id, dst_net_node_id)
                    y_var_key = EdgeAssKey(mod_edge_id, net_edge_id)
                    y_var = edge_ass_vars[y_var_key]

                    net_node_sum_elems.append(
                        y_var
                        * self.__get_transmission_time(
                            model_graph.get_edge_info(mod_edge_id),
                            network_graph.get_edge_info(net_edge_id),
                            net_edge_id,
                        )
                    )

            net_node_sum = pulp.lpSum(net_node_sum_elems)
            tot_sum_elems.append(
                net_node_sum
                * network_graph.get_node_info(net_node_id).get_info(
                    GraphInfo.NET_NODE_TRANS_ENERGY_PER_SEC
                )
            )

        return pulp.lpSum(tot_sum_elems)

    def define_node_assignment_vars(
        self,
        problem: pulp.LpProblem,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
    ) -> dict[NodeAssKey, pulp.LpVariable]:

        vars_table: dict[NodeAssKey, pulp.LpVariable] = {}
        for mod_node_id in model_graph.get_nodes_id():
            for net_node_id in network_graph.get_nodes_id():

                var_name: str = self.__build_assignment_var_name(
                    mod_node_id, net_node_id
                )
                lp_variable = pulp.LpVariable(var_name, cat=pulp.LpBinary)

                # problem.addVariable(lp_variable)
                table_key = NodeAssKey(mod_node_id, net_node_id)
                vars_table[table_key] = lp_variable

        return vars_table

    def define_edge_assignment_vars(
        self,
        problem: pulp.LpProblem,
        model_graph: ModelGraph,
        network_graph: NetworkGraph,
    ) -> dict[EdgeAssKey, pulp.LpVariable]:
        vars_table: dict[EdgeAssKey, pulp.LpVariable] = {}
        for mod_edge_id in model_graph.get_edges_id():
            for net_edge_id in network_graph.get_edges_id():

                var_name: str = self.__build_edge_var_name(mod_edge_id, net_edge_id)
                lp_variable = pulp.LpVariable(var_name, cat=pulp.LpBinary)

                table_key = EdgeAssKey(mod_edge_id, net_edge_id)
                vars_table[table_key] = lp_variable

        return vars_table

    def __build_assignment_var_name(self, modelNode: NodeId, networkNode: NodeId):
        return "x_(mod_node_{})(net_node_{})".format(modelNode, networkNode)

    def __build_edge_var_name(self, modelEdge: EdgeId, networkEdge: EdgeId):
        return "y_(mod_node_{})(net_node_{})".format(modelEdge, networkEdge)

    def __get_transmission_time(
        self,
        mod_edge_info: ModelEdgeInfo,
        net_edge_info: NetworkEdgeInfo,
        net_edge_id: EdgeId,
    ) -> float:
        ## Note --> Assuming Bandwidth in Byte / s

        if net_edge_id.first_node_id == net_edge_id.second_node_id:
            return 0

        return (
            mod_edge_info.get_model_edge_data_size()
            / net_edge_info.get_edge_bandwidth()
        )

    def __get_computation_time(
        self, mod_node_info: ModelNodeInfo, net_node_info: NetworkNodeInfo
    ) -> float:
        return mod_node_info.get_node_flops() / net_node_info.get_flops_per_sec()
