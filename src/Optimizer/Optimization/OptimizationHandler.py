import time
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
    latency_weight: float
    energy_weight: float

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

        start = time.perf_counter_ns()
        print("Building Optimization Problem")
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

            ConstraintsBuilder.add_memory_constraints(
                problem,
                curr_mod_graph,
                network_graph,
                node_ass_vars,
                mem_use_vars,
                opt_params.requests_number.get(curr_mod_graph.get_graph_name()),
            )

            ## TODO Implement this one
            ConstraintsBuilder.add_energy_constraints()

        ## Computing Latency Objective
        latency_cost = LatencyComputer.compute_latency_cost(
            model_graphs,
            network_graph,
            node_ass_vars,
            edge_ass_vars,
            opt_params.requests_number,
        )

        ## Computing Energy Objective
        energy_cost = EnergyComputer.compute_energy_cost(
            model_graphs,
            network_graph,
            node_ass_vars,
            edge_ass_vars,
            opt_params.requests_number,
        )

        problem += (
            opt_params.latency_weight * latency_cost
            + opt_params.energy_weight * energy_cost
        )

        end = time.perf_counter_ns()
        print("Building Optimization Problem Time >> ", (end - start) / 1e9)

        problem.solve(pulp.GLPK_CMD())

        with open("VarFile.txt", "w") as f:
            for var in problem.variables():
                f.write(f"{var.name} = {var.varValue}\n")

        # Print the objective function value
        print(f"Objective value = {pulp.value(problem.objective)}")

        problem.writeLP("solved_problem.lp")

        solved_info_dict: dict[str, SolvedProblemInfo] = {}
        for mod_graph in model_graphs:
            start = time.perf_counter_ns()
            print("Building Solution >> ", mod_graph.get_graph_name())
            solved_problem_info: SolvedProblemInfo = self.build_solved_problem_info(
                problem, mod_graph.get_graph_name(), node_ass_vars, edge_ass_vars
            )
            solved_info_dict[mod_graph.get_graph_name()] = solved_problem_info
            end = time.perf_counter_ns()
            print("Building Solution Time >> ", (end - start) / 1e9)
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

        filtered_node_ass = dict(
            filter(lambda item: item[0].mod_name == graph_name, node_ass_vars.items())
        )
        filtered_edge_ass = dict(
            filter(lambda item: item[0].mod_name == graph_name, edge_ass_vars.items())
        )

        for node_ass_key, node_ass_var in filtered_node_ass.items():
            if node_ass_var.varValue == 1.0:
                mod_node_id = node_ass_key.mod_node_id
                net_node_id = node_ass_key.net_node_id
                solved_problem_info.put_node_assignment(net_node_id, mod_node_id)
                pass
            pass

        for edge_ass_key, edge_ass_var in filtered_edge_ass.items():
            if edge_ass_var.varValue == 1.0:
                mod_edge_id = edge_ass_key.mod_edge_id
                net_edge_id = edge_ass_key.net_edge_id
                solved_problem_info.put_edge_assignment(net_edge_id, mod_edge_id)
                pass
            pass

        return solved_problem_info
