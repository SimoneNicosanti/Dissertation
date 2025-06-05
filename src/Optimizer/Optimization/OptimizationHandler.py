import time
from dataclasses import dataclass

import networkx as nx
import pulp

from CommonIds.NodeId import NodeId
from CommonProfile.ExecutionProfile import ServerExecutionProfilePool
from CommonProfile.ModelInfo import ModelEdgeInfo, ModelNodeInfo
from CommonProfile.ModelProfile import ModelProfile, Regressor
from Optimizer.Optimization import EnergyComputer, LatencyComputer, VarsBuilder
from Optimizer.Optimization.ConstraintsBuilder import ConstraintsBuilder
from Optimizer.Optimization.OptimizationKeys import (
    EdgeAssKey,
    MemoryUseKey,
    NodeAssKey,
    QuantizationKey,
)
from Optimizer.Optimization.RegressorBuilder import RegressorBuilder


@dataclass
class OptimizationParams:
    latency_weight: float
    energy_weight: float

    device_max_energy: float

    requests_number: dict[str, int]

    max_noises: dict[str, float]


class OptimizationHandler:

    def __init__(self):
        pass

    @staticmethod
    def optimize(
        model_profile_list: list[ModelProfile],
        network_graph: nx.DiGraph,
        start_server: NodeId,
        opt_params: OptimizationParams = None,
        server_execution_profile_pool: ServerExecutionProfilePool = None,
    ) -> list[nx.DiGraph]:

        model_graphs: list[nx.DiGraph] = [
            model_profile.get_model_graph() for model_profile in model_profile_list
        ]

        problem: pulp.LpProblem = pulp.LpProblem("Partitioning", pulp.LpMinimize)

        node_ass_vars: dict[NodeAssKey, pulp.LpVariable] = {}
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable] = {}
        mem_use_vars: dict[MemoryUseKey, pulp.LpVariable] = {}
        quantization_vars: dict[QuantizationKey, pulp.LpVariable] = {}

        time.perf_counter_ns()
        ## Defining variables
        for curr_mod_graph in model_graphs:
            curr_node_ass_vars: dict[NodeAssKey, pulp.LpVariable] = (
                VarsBuilder.define_node_assignment_vars(curr_mod_graph, network_graph)
            )
            node_ass_vars.update(curr_node_ass_vars)
            print("Done Defining Node Ass Vars")

            curr_edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable] = (
                VarsBuilder.define_edge_assignment_vars(curr_mod_graph, network_graph)
            )
            edge_ass_vars.update(curr_edge_ass_vars)
            print("Done Defining Edge Ass Vars")

            curr_mem_use_vars: dict[MemoryUseKey, pulp.LpVariable] = (
                VarsBuilder.define_memory_use_vars(
                    network_graph, curr_mod_graph.graph["name"]
                )
            )
            mem_use_vars.update(curr_mem_use_vars)
            print("Done Defining Memory Usage Vars")

            curr_quant_vars: dict[QuantizationKey, pulp.LpVariable] = (
                VarsBuilder.define_quantization_vars(curr_mod_graph)
            )
            quantization_vars.update(curr_quant_vars)
            print("Done Defining Quantization Vars")

        ## Adding Constraints
        for curr_mod_graph in model_graphs:
            ConstraintsBuilder.add_node_assignment_constraints(
                problem, curr_mod_graph, node_ass_vars, start_server
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

        ConstraintsBuilder.add_node_ass_vars_product_constraint(
            problem, node_ass_vars, quantization_vars
        )
        ConstraintsBuilder.add_edge_ass_vars_product_constraint(
            problem, edge_ass_vars, quantization_vars
        )

        ConstraintsBuilder.add_memory_constraints(
            problem,
            model_graphs,
            network_graph,
            node_ass_vars,
            mem_use_vars,
        )

        if opt_params.device_max_energy > 0:
            ## If <= 0 we assume no boundary
            ConstraintsBuilder.add_energy_constraints(
                problem,
                model_graphs,
                network_graph,
                node_ass_vars,
                edge_ass_vars,
                opt_params.requests_number,
                start_server,
                opt_params.device_max_energy,
            )

        for model_profile in model_profile_list:
            regressor: Regressor = model_profile.get_regressor()

            regressor_expression = RegressorBuilder.build_regressor_expression(
                problem, regressor, model_profile.get_model_graph(), quantization_vars
            )

            problem += (
                regressor_expression
                <= opt_params.max_noises[model_profile.get_model_name()]
            )

        ## Computing Latency Objective
        latency_cost = LatencyComputer.compute_latency_cost(
            model_graphs,
            network_graph,
            node_ass_vars,
            edge_ass_vars,
            opt_params.requests_number,
            server_execution_profile_pool,
        )

        ## Computing Energy Objective
        energy_cost = EnergyComputer.compute_energy_cost(
            model_graphs,
            network_graph,
            node_ass_vars,
            edge_ass_vars,
            opt_params.requests_number,
            server_execution_profile_pool,
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

        for quant_key in quantization_vars.keys():
            print(quant_key.mod_node_id, quantization_vars[quant_key].varValue)
        print(regressor_expression.value())

        if pulp.LpStatus[problem.status] != "Optimal":
            return None

        with open("/optimizer_data/plans/solved.lp", "w") as _:
            problem.writeLP("/optimizer_data/plans/solved.lp")

        print("Solved With Cost >> ", problem.objective.value())
        solved_model_graphs: list[nx.DiGraph] = []
        for mod_graph in model_graphs:
            time.perf_counter_ns()
            solved_model_graph: nx.DiGraph = (
                OptimizationHandler.build_solved_model_graph(
                    problem, mod_graph, node_ass_vars, edge_ass_vars
                )
            )
            solved_model_graphs.append(solved_model_graph)
            time.perf_counter_ns()
        return solved_model_graphs

    def build_solved_model_graph(
        problem: pulp.LpProblem,
        model_graph: nx.DiGraph,
        node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
        edge_ass_vars: dict[EdgeAssKey, pulp.LpVariable],
    ) -> nx.DiGraph:

        graph_name = model_graph.graph["name"]
        if pulp.LpStatus[problem.status] != pulp.LpStatus[pulp.LpStatusOptimal]:
            ## Problem could not be solved
            return nx.DiGraph(name=graph_name, solved=False, value=float("inf"))

        solved_model_graph = nx.DiGraph(
            name=graph_name, solved=True, value=problem.objective.value()
        )

        filtered_node_ass: dict[NodeAssKey, pulp.LpVariable] = dict(
            filter(
                lambda item: item[0].mod_name == graph_name
                and not item[0].is_quantized,
                node_ass_vars.items(),
            )
        )
        filtered_edge_ass: dict[EdgeAssKey, pulp.LpVariable] = dict(
            filter(
                lambda item: item[0].mod_name == graph_name
                and not item[0].is_quantized,
                edge_ass_vars.items(),
            )
        )

        for node_ass_key, node_ass_var in filtered_node_ass.items():
            if node_ass_var.varValue == 1.0:
                mod_node_id = node_ass_key.mod_node_id
                net_node_id = node_ass_key.net_node_id

                solved_model_graph.add_node(
                    mod_node_id,
                    net_node_id=net_node_id,
                    generator=model_graph.nodes[mod_node_id].get(
                        ModelNodeInfo.GENERATOR, False
                    ),
                    receiver=model_graph.nodes[mod_node_id].get(
                        ModelNodeInfo.RECEIVER, False
                    ),
                )

                # if model_graph.nodes[mod_node_id].get(ModelNodeInfo.QUANTIZABLE, False):
                #     quant_node_ass_key = NodeAssKey(
                #         mod_node_id, net_node_id, graph_name, True
                #     )
                #     quant_node_ass_var = node_ass_vars[quant_node_ass_key]

            pass

        for edge_ass_key, edge_ass_var in filtered_edge_ass.items():
            if edge_ass_var.varValue == 1.0:
                mod_edge_id = edge_ass_key.mod_edge_id
                net_edge_id = edge_ass_key.net_edge_id

                solved_model_graph.add_edge(
                    mod_edge_id[0],
                    mod_edge_id[1],
                    net_edge_id=net_edge_id,
                    tensor_name_list=model_graph.edges[mod_edge_id].get(
                        ModelEdgeInfo.TENSOR_NAME_LIST, []
                    ),
                )

            pass

        return solved_model_graph
