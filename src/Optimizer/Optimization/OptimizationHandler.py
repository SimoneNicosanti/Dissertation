import time
from dataclasses import dataclass

import networkx as nx
import pulp

from CommonIds.NodeId import NodeId
from CommonPlan.SolvedModelGraph import SolvedNodeInfo
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

        (
            first_problem,
            first_node_ass_vars,
            first_edge_ass_vars,
            first_quantization_vars,
            _,
            _,
        ) = OptimizationHandler.prepare_problem(
            model_profile_list,
            network_graph,
            start_server,
            opt_params,
            server_execution_profile_pool,
        )

        latency_weight = opt_params.latency_weight / (
            opt_params.latency_weight + opt_params.energy_weight
        )
        energy_weight = opt_params.energy_weight / (
            opt_params.latency_weight + opt_params.energy_weight
        )

        if latency_weight >= energy_weight:
            ## First optimize Latency
            first_problem_cost = LatencyComputer.compute_latency_cost(
                model_graphs,
                network_graph,
                first_node_ass_vars,
                first_edge_ass_vars,
                opt_params.requests_number,
                server_execution_profile_pool,
            )

            pass
        else:
            ## First optimize Energy
            first_problem_cost = EnergyComputer.compute_energy_cost(
                model_graphs,
                network_graph,
                first_node_ass_vars,
                first_edge_ass_vars,
                opt_params.requests_number,
                server_execution_profile_pool,
            )

            pass

        first_problem += first_problem_cost
        first_problem.solve(pulp.GLPK_CMD(msg=False))

        if pulp.LpStatus[first_problem.status] != "Optimal":
            return None

        first_optimum_value = first_problem.objective.value()
        print("First Optimization Value: ", first_optimum_value)

        max_weight = max(latency_weight, energy_weight)
        allowed_increase = 1 - max_weight
        print("Allowed Increase: ", allowed_increase)

        (
            final_problem,
            final_node_ass_vars,
            final_edge_ass_vars,
            final_quantization_vars,
            device_max_exergy,
            regressors_expr,
        ) = OptimizationHandler.prepare_problem(
            model_profile_list,
            network_graph,
            start_server,
            opt_params,
            server_execution_profile_pool,
        )
        print("Done Defining Final Problem")

        if latency_weight >= energy_weight:
            ## Optimize Energy Binding Latency

            final_other_cost = LatencyComputer.compute_latency_cost(
                model_graphs,
                network_graph,
                final_node_ass_vars,
                final_edge_ass_vars,
                opt_params.requests_number,
                server_execution_profile_pool,
            )

            final_problem_cost = EnergyComputer.compute_energy_cost(
                model_graphs,
                network_graph,
                final_node_ass_vars,
                final_edge_ass_vars,
                opt_params.requests_number,
                server_execution_profile_pool,
            )

            pass
        else:
            ## Optimize Latency Binding Energy

            final_other_cost = EnergyComputer.compute_energy_cost(
                model_graphs,
                network_graph,
                final_node_ass_vars,
                final_edge_ass_vars,
                opt_params.requests_number,
                server_execution_profile_pool,
            )

            final_problem_cost = LatencyComputer.compute_latency_cost(
                model_graphs,
                network_graph,
                final_node_ass_vars,
                final_edge_ass_vars,
                opt_params.requests_number,
                server_execution_profile_pool,
            )

        final_problem += final_other_cost <= first_optimum_value * (
            1 + allowed_increase
        )
        final_problem += final_problem_cost

        final_problem.solve(pulp.GLPK_CMD(msg=False))

        if pulp.LpStatus[final_problem.status] != "Optimal":
            return None

        first_obj_name = "⌛ Latency" if latency_weight >= energy_weight else "Energy"
        second_obj_name = "🔋 Energy" if latency_weight >= energy_weight else "Latency"
        print(f"Final {first_obj_name} Optimized Value: ", final_other_cost.value())
        print(f"Final {second_obj_name} Optimized Value: ", final_problem_cost.value())
        print(
            "Final 🪫 Device Energy: ",
            None if device_max_exergy is None else device_max_exergy.value(),
        )
        for regressor in regressors_expr:
            print("Final 📈 Regressor Value:", regressor.value())

        solved_model_graphs: list[nx.DiGraph] = []
        for mod_graph in model_graphs:
            time.perf_counter_ns()
            solved_model_graph: nx.DiGraph = (
                OptimizationHandler.build_solved_model_graph(
                    final_problem,
                    mod_graph,
                    final_node_ass_vars,
                    final_edge_ass_vars,
                    final_quantization_vars,
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
        quantization_vars: dict[QuantizationKey, pulp.LpVariable] = None,
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
        filtered_quant_vars: dict[QuantizationKey, pulp.LpVariable] = dict(
            filter(
                lambda item: item[0].mod_name == graph_name,
                quantization_vars.items(),
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

            pass

        for edge_ass_key, edge_ass_var in filtered_edge_ass.items():
            if edge_ass_var.varValue == 1.0:
                mod_edge_id = edge_ass_key.mod_edge_id
                net_edge_id = edge_ass_key.net_edge_id

                ## Renaming tensors according to quantization
                ## TODO This would be better placed in other class (like plan builder)
                quantization_key = QuantizationKey(
                    mod_edge_id[0], edge_ass_key.mod_name
                )
                tensor_name_list = []
                if (
                    quantization_key in quantization_vars
                    and quantization_vars[quantization_key].varValue == 1.0
                ):
                    for elem in model_graph.edges[mod_edge_id].get(
                        ModelEdgeInfo.TENSOR_NAME_LIST, []
                    ):
                        renamed_elem = elem + "_QuantizeLinear_Output"
                        tensor_name_list.append(renamed_elem)
                        pass
                else:
                    tensor_name_list = model_graph.edges[mod_edge_id].get(
                        ModelEdgeInfo.TENSOR_NAME_LIST, []
                    )

                solved_model_graph.add_edge(
                    mod_edge_id[0],
                    mod_edge_id[1],
                    net_edge_id=net_edge_id,
                    tensor_name_list=tensor_name_list,
                )

            pass

        for quant_key, quant_var in filtered_quant_vars.items():
            if quant_var.varValue == 1.0:

                mod_node_id = quant_key.mod_node_id
                solved_model_graph.nodes[mod_node_id][SolvedNodeInfo.QUANTIZED] = True

        return solved_model_graph

    @staticmethod
    def prepare_problem(
        model_profile_list: list[ModelProfile],
        network_graph: nx.DiGraph,
        start_server: NodeId,
        opt_params: OptimizationParams = None,
        server_execution_profile_pool: ServerExecutionProfilePool = None,
    ):
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

        device_energy_expr = None
        if opt_params.device_max_energy > 0:
            ## If <= 0 we assume no boundary
            device_energy_expr = ConstraintsBuilder.add_energy_constraints(
                problem,
                model_graphs,
                network_graph,
                node_ass_vars,
                edge_ass_vars,
                opt_params.requests_number,
                start_server,
                opt_params.device_max_energy,
                server_execution_profile_pool,
            )

        regressor_expressions = []
        for model_profile in model_profile_list:
            regressor: Regressor = model_profile.get_regressor()

            regressor_expression = RegressorBuilder.build_regressor_expression(
                problem, regressor, model_profile.get_model_graph(), quantization_vars
            )

            problem += (
                regressor_expression
                <= opt_params.max_noises[model_profile.get_model_name()]
            )

            regressor_expressions.append(regressor_expression)

        return (
            problem,
            node_ass_vars,
            edge_ass_vars,
            quantization_vars,
            device_energy_expr,
            regressor_expressions,
        )
