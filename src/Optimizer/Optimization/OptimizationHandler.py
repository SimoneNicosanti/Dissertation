import time
from dataclasses import dataclass

import pulp

from Optimizer.Graph.Graph import NodeId
from Optimizer.Graph.ModelGraph import ModelGraph
from Optimizer.Graph.NetworkGraph import NetworkGraph
from Optimizer.Graph.SolvedModelGraph import (
    SolvedEdgeInfo,
    SolvedModelGraph,
    SolvedNodeInfo,
)
from Optimizer.Optimization import EnergyComputer, LatencyComputer, VarsBuilder
from Optimizer.Optimization.ConstraintsBuilder import ConstraintsBuilder
from Optimizer.Optimization.OptimizationKeys import EdgeAssKey, MemoryUseKey, NodeAssKey


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
    ) -> list[SolvedModelGraph]:
        problem: pulp.LpProblem = pulp.LpProblem("Partitioning", pulp.LpMinimize)

        node_ass_vars: dict[NodeAssKey, pulp.LpVariable] = {}
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable] = {}
        mem_use_vars: dict[MemoryUseKey, pulp.LpVariable] = {}

        time.perf_counter_ns()
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
            model_graphs,
            network_graph,
            node_ass_vars,
            mem_use_vars,
            opt_params.requests_number.get(curr_mod_graph.get_graph_name()),
        )

        ## TODO Activate this when known energy model
        # ConstraintsBuilder.add_energy_constraints(
        #     problem,
        #     model_graphs,
        #     network_graph,
        #     node_ass_vars,
        #     opt_params.requests_number,
        #     deployment_server,
        #     opt_params.device_max_energy,
        # )

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

        latency_weight = opt_params.latency_weight / (
            opt_params.latency_weight + opt_params.energy_weight
        )
        energy_weight = opt_params.energy_weight / (
            opt_params.latency_weight + opt_params.energy_weight
        )

        problem += latency_weight * latency_cost + energy_weight * energy_cost

        time.perf_counter_ns()

        problem.solve(pulp.GLPK_CMD())

        with open("./Optimizer/solved_problem/VarFile.txt", "w") as f:
            for var in problem.variables():
                f.write(f"{var.name} = {var.varValue}\n")

        # Print the objective function value
        print(f"Objective value = {pulp.value(problem.objective)}")

        # problem.writeLP("./solved_problem/solved_problem.lp")

        solved_model_graphs: list[SolvedModelGraph] = []
        for mod_graph in model_graphs:
            time.perf_counter_ns()
            solved_model_graph: SolvedModelGraph = self.build_solved_model_graph(
                problem, mod_graph, node_ass_vars, edge_ass_vars
            )
            solved_model_graphs.append(solved_model_graph)
            time.perf_counter_ns()
        return solved_model_graphs

    def build_solved_model_graph(
        self,
        problem: pulp.LpProblem,
        model_graph: ModelGraph,
        node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    ) -> SolvedModelGraph:

        graph_name = model_graph.get_graph_name()
        if pulp.LpStatus[problem.status] != pulp.LpStatus[pulp.LpStatusOptimal]:
            ## Problem could not be solved
            return SolvedModelGraph(
                graph_name=graph_name, problem_solved=False, solution_value=float("inf")
            )

        solved_model_graph = SolvedModelGraph(
            graph_name, problem_solved=True, solution_value=problem.objective.value()
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

                node_info = SolvedNodeInfo(
                    net_node_id,
                    ModelGraph.is_generator_node(mod_node_id),
                    ModelGraph.is_receiver_node(mod_node_id),
                )
                solved_model_graph.put_node(mod_node_id, node_info)
                pass
            pass

        for edge_ass_key, edge_ass_var in filtered_edge_ass.items():
            if edge_ass_var.varValue == 1.0:
                mod_edge_id = edge_ass_key.mod_edge_id
                net_edge_id = edge_ass_key.net_edge_id

                tensor_names = model_graph.get_edge_info(mod_edge_id).get_tensor_names()

                edge_info = SolvedEdgeInfo(net_edge_id, tensor_names)
                solved_model_graph.put_edge(mod_edge_id, edge_info)
                pass
            pass

        return solved_model_graph
