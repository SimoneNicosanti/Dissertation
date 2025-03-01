from dataclasses import dataclass

import pulp
from Graph.Graph import NodeId
from Graph.ModelGraph import ModelGraph
from Graph.NetworkGraph import NetworkGraph
from Optimization import EnergyComputer, LatencyComputer, VarsBuilder
from Optimization.ConstraintsBuilder import ConstraintsBuilder
from Optimization.OptimizationKeys import EdgeAssKey, MemoryUseKey, NodeAssKey
from Optimization.SolvedProblemInfo import SolvedProblemInfo


@dataclass
class OptimizationParams:
    comp_lat_weight: float
    trans_lat_weight: float
    comp_en_weight: float
    trans_en_weight: float

    device_max_energy: float

    requests_number: dict[str, int]


class OptimizationHandler:

    def __init__(self):
        pass

    def optimize(
        self,
        model_graphs: list[ModelGraph],
        network_graph: NetworkGraph,
        deployment_server: NodeId,
        opt_params: OptimizationParams = None,
    ) -> dict[str, SolvedProblemInfo]:
        problem: pulp.LpProblem = pulp.LpProblem("Partitioning", pulp.LpMinimize)

        node_ass_vars: dict[NodeAssKey, pulp.LpVariable] = {}
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable] = {}
        mem_use_vars: dict[MemoryUseKey, pulp.LpVariable] = {}

        ## Defining variables
        for curr_mod_graph in model_graphs:
            curr_node_ass_vars: dict[NodeAssKey, pulp.LpVariable] = (
                VarsBuilder.define_node_assignment_vars(curr_mod_graph, network_graph)
            )
            node_ass_vars.update(curr_node_ass_vars)

            curr_edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable] = (
                VarsBuilder.define_edge_assignment_vars(curr_mod_graph, network_graph)
            )
            edge_ass_vars.update(curr_edge_ass_vars)

            curr_mem_use_vars: dict[MemoryUseKey, pulp.LpVariable] = (
                VarsBuilder.define_memory_use_vars(
                    network_graph, curr_mod_graph.get_graph_name()
                )
            )
            mem_use_vars.update(curr_mem_use_vars)

        ## Adding Constraints
        for curr_mod_graph in model_graphs:
            ConstraintsBuilder.add_node_assignment_constraints(
                problem, curr_mod_graph, node_ass_vars, deployment_server
            )
            ConstraintsBuilder.add_edge_assignment_constraints(
                problem, curr_mod_graph, edge_ass_vars
            )

            ConstraintsBuilder.add_input_flow_constraints(
                problem, curr_mod_graph, network_graph, node_ass_vars, edge_ass_vars
            )
            ConstraintsBuilder.add_output_flow_constraints(
                problem, curr_mod_graph, network_graph, node_ass_vars, edge_ass_vars
            )

            ## TODO Implement these two
            ConstraintsBuilder.add_memory_constraints(
                problem,
                curr_mod_graph,
                network_graph,
                node_ass_vars,
                mem_use_vars,
                opt_params.requests_number.get(curr_mod_graph.get_graph_name()),
            )
            ConstraintsBuilder.add_energy_constraints()

        ## Computing Ltency Objective
        comp_latency, trans_latency = LatencyComputer.find_latency_component(
            model_graphs,
            network_graph,
            node_ass_vars,
            edge_ass_vars,
            opt_params.requests_number,
        )

        ## Computing Energy Objective
        comp_energy, trans_energy = EnergyComputer.find_energy_component(
            model_graphs,
            network_graph,
            node_ass_vars,
            edge_ass_vars,
            opt_params.requests_number,
        )

        problem += (
            opt_params.comp_lat_weight * comp_latency
            + opt_params.trans_lat_weight * trans_latency
            + opt_params.comp_en_weight * comp_energy
            + opt_params.trans_lat_weight * trans_energy
        )

        problem.solve(pulp.GLPK_CMD())

        with open("VarFile.txt", "w") as f:
            for var in problem.variables():
                f.write(f"{var.name} = {var.varValue}\n")

        # Print the objective function value
        print(f"Objective value = {pulp.value(problem.objective)}")

        problem.writeLP("solved_problem.lp")

        solved_info_dict: dict[str, SolvedProblemInfo] = {}
        for mod_graph in model_graphs:
            solved_problem_info: SolvedProblemInfo = self.build_solved_problem_info(
                problem, mod_graph.get_graph_name(), node_ass_vars, edge_ass_vars
            )
            solved_info_dict[mod_graph.get_graph_name()] = solved_problem_info

        return solved_info_dict

    def build_solved_problem_info(
        self,
        problem: pulp.LpProblem,
        graph_name: str,
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
                    if (
                        var.name == node_ass_var.name
                        and var.varValue == 1.0
                        and node_ass_key.mod_name == graph_name
                    ):
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
                    if (
                        var.name == edge_ass_var.name
                        and var.varValue == 1.0
                        and edge_ass_key.mod_name == graph_name
                    ):
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
