import os
import time
from dataclasses import dataclass

import networkx as nx
import pulp

from CommonIds.NodeId import NodeId
from CommonPlan.SolvedModelGraph import SolvedGraphInfo, SolvedNodeInfo
from CommonProfile.ExecutionProfile import ServerExecutionProfilePool
from CommonProfile.ModelInfo import ModelEdgeInfo, ModelGraphInfo, ModelNodeInfo
from CommonProfile.ModelProfile import ModelProfile, Regressor
from Optimizer.Optimization import EnergyComputer, LatencyComputer, VarsBuilder
from Optimizer.Optimization.ConstraintsBuilder import ConstraintsBuilder
from Optimizer.Optimization.OptimizationKeys import (
    MemoryUseKey,
    NodeAssKey,
    QuantizationKey,
    TensorAssKey,
)
from Optimizer.Optimization.RegressorBuilder import RegressorBuilder

INT_VARS_VALUE_THR = 0.90  ## To handle float approximation
CPLEX_PATH = "/opt/ibm/cplex/cplex/bin/x86-64_linux/cplex"


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
    def get_solver(warmStart=False, gapRel=None):
        if os.path.exists(CPLEX_PATH):
            return pulp.CPLEX_CMD(
                path=CPLEX_PATH, warmStart=False, gapRel=gapRel, msg=False
            )
        else:
            return pulp.SCIP_PY(warmStart=warmStart, msg=False)

    def prepare_warm_start(
        src_node_ass_vars,
        src_tensor_ass_vars,
        src_quantization_vars,
        dst_node_ass_vars,
        dst_tensor_ass_vars,
        dst_quantization_vars,
    ):
        return

        for key in src_node_ass_vars:
            dst_node_ass_vars[key].setInitialValue(
                1 if src_node_ass_vars[key] >= INT_VARS_VALUE_THR else 0
            )
        for key in src_tensor_ass_vars:
            dst_tensor_ass_vars[key].setInitialValue(
                1 if src_tensor_ass_vars[key] >= INT_VARS_VALUE_THR else 0
            )
        for key in src_quantization_vars:
            dst_quantization_vars[key].setInitialValue(
                1 if src_quantization_vars[key] >= INT_VARS_VALUE_THR else 0
            )

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

        problem: pulp.LpProblem
        (
            problem,
            node_ass_vars,
            tensor_ass_vars,
            quantization_vars,
            device_energy_expr,
            regr_expr,
        ) = OptimizationHandler.prepare_problem(
            model_profile_list,
            network_graph,
            start_server,
            opt_params,
            server_execution_profile_pool,
        )

        min_latency_cost = LatencyComputer.compute_latency_cost(
            model_graphs,
            network_graph,
            node_ass_vars,
            tensor_ass_vars,
            opt_params.requests_number,
            server_execution_profile_pool,
        )

        problem.setObjective(min_latency_cost)
        problem.sense = pulp.LpMinimize
        problem.solve(OptimizationHandler.get_solver(gapRel=0.01))
        if pulp.LpStatus[problem.status] != "Optimal":
            return None
        min_lat_cost = min_latency_cost.value()

        en_val_on_min_lat = EnergyComputer.compute_energy_cost(
            model_graphs,
            network_graph,
            node_ass_vars,
            tensor_ass_vars,
            opt_params.requests_number,
            server_execution_profile_pool,
        )
        en_val_on_min_lat = en_val_on_min_lat.value()

        min_energy_cost = EnergyComputer.compute_energy_cost(
            model_graphs,
            network_graph,
            node_ass_vars,
            tensor_ass_vars,
            opt_params.requests_number,
            server_execution_profile_pool,
        )
        problem.setObjective(min_energy_cost)
        problem.sense = pulp.LpMinimize
        problem.solve(OptimizationHandler.get_solver(gapRel=0.01))
        if pulp.LpStatus[problem.status] != "Optimal":
            return None
        min_en_cost = min_energy_cost.value()

        lat_val_on_min_en = LatencyComputer.compute_latency_cost(
            model_graphs,
            network_graph,
            node_ass_vars,
            tensor_ass_vars,
            opt_params.requests_number,
            server_execution_profile_pool,
        )
        lat_val_on_min_en = lat_val_on_min_en.value()

        max_lat_cost = max(lat_val_on_min_en, min_lat_cost)
        max_en_cost = max(en_val_on_min_lat, min_en_cost)

        final_latency_cost = LatencyComputer.compute_latency_cost(
            model_graphs,
            network_graph,
            node_ass_vars,
            tensor_ass_vars,
            opt_params.requests_number,
            server_execution_profile_pool,
        )
        final_latency_cost = (final_latency_cost - min_lat_cost) / (
            max_lat_cost - min_lat_cost + 1e-9
        )
        print("Max and Min Latency Cost >> ", max_lat_cost, min_lat_cost)

        final_energy_cost = EnergyComputer.compute_energy_cost(
            model_graphs,
            network_graph,
            node_ass_vars,
            tensor_ass_vars,
            opt_params.requests_number,
            server_execution_profile_pool,
        )
        final_energy_cost = (final_energy_cost - min_en_cost) / (
            max_en_cost - min_en_cost + 1e-9
        )
        print("Max and Min Energy Cost >> ", max_en_cost, min_en_cost)

        latency_weight = opt_params.latency_weight / (
            opt_params.latency_weight + opt_params.energy_weight
        )
        energy_weight = opt_params.energy_weight / (
            opt_params.latency_weight + opt_params.energy_weight
        )

        problem.setObjective(
            latency_weight * final_latency_cost + energy_weight * final_energy_cost
        )
        problem.sense = pulp.LpMinimize
        problem.solve(OptimizationHandler.get_solver())

        if pulp.LpStatus[problem.status] != "Optimal":
            return None

        print("💰 Latency Cost >> ", final_latency_cost.value())
        print("💡 Energy Cost >> ", final_energy_cost.value())

        latency_value = (
            final_latency_cost.value() * (max_lat_cost - min_lat_cost + 1e-9)
            + min_lat_cost
        )
        energy_value = (
            final_energy_cost.value() * (max_en_cost - min_en_cost + 1e-9) + min_en_cost
        )

        print("⌛ Final Latency Value  >> ", latency_value)
        print("🔋 Final Energy Value  >> ", energy_value)
        print("🪫 Device Energy Value  >> ", device_energy_expr.value())
        print("📈 Regressor Value  >> ", regr_expr[0].value())

        solved_model_graphs: list[nx.DiGraph] = []
        for mod_graph in model_graphs:
            time.perf_counter_ns()
            solved_model_graph: nx.DiGraph = (
                OptimizationHandler.build_solved_model_graph(
                    problem,
                    mod_graph,
                    node_ass_vars,
                    tensor_ass_vars,
                    quantization_vars,
                )
            )

            ## TODO We are considering only one model: it is ok for now
            solved_model_graph.graph[SolvedGraphInfo.LATENCY_VALUE] = latency_value
            solved_model_graph.graph[SolvedGraphInfo.ENERGY_VALUE] = energy_value
            solved_model_graph.graph[SolvedGraphInfo.DEVICE_ENERGY_VALUE] = (
                device_energy_expr.value()
            )
            solved_model_graph.graph[SolvedGraphInfo.LATENCY_COST] = (
                final_latency_cost.value()
            )
            solved_model_graph.graph[SolvedGraphInfo.ENERGY_COST] = (
                final_energy_cost.value()
            )

            solved_model_graphs.append(solved_model_graph)
            time.perf_counter_ns()

        return solved_model_graphs

    def build_solved_model_graph(
        problem: pulp.LpProblem,
        model_graph: nx.DiGraph,
        node_ass_vars: dict[NodeAssKey, pulp.LpVariable],
        tensor_ass_vars: dict[TensorAssKey, pulp.LpVariable],
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
        filtered_tensor_ass: dict[TensorAssKey, pulp.LpVariable] = dict(
            filter(
                lambda item: item[0].mod_name == graph_name
                and not item[0].is_quantized,
                tensor_ass_vars.items(),
            )
        )
        filtered_quant_vars: dict[QuantizationKey, pulp.LpVariable] = dict(
            filter(
                lambda item: item[0].mod_name == graph_name,
                quantization_vars.items(),
            )
        )

        assigned_nodes_dict: dict[NodeId, set[NodeId]] = {}

        print("Analyzing node assignments...")
        tot_assigned = 0
        for node_ass_key, node_ass_var in filtered_node_ass.items():
            if node_ass_var.varValue >= INT_VARS_VALUE_THR:
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

                assigned_nodes_dict.setdefault(net_node_id, set())
                assigned_nodes_dict[net_node_id].add(mod_node_id)

                tot_assigned += 1

            pass
        print("Done Analyzing node assignments...")
        print("\t Total assigned nodes: ", tot_assigned)

        print("Analyzing tensor assignments...")
        tensors_dict: dict[str, list] = model_graph.graph[
            ModelGraphInfo.TENSOR_SIZE_DICT
        ]
        all_dests_per_tensor: dict[str, set[NodeId]] = {}
        for edge in model_graph.edges:
            for tensor_name in tensors_dict.keys():
                all_dests_per_tensor.setdefault(tensor_name, set())
                if tensor_name in model_graph.edges[edge].get(
                    ModelEdgeInfo.TENSOR_NAME_LIST
                ):
                    all_dests_per_tensor[tensor_name].add(edge[1])
        print("\t Done Analyzing tensors dests...")

        print("\t Analyzing actual tensor assignments...")
        for tensor_ass_key, tensor_ass_var in filtered_tensor_ass.items():
            tensor_name = tensor_ass_key.tensor_name
            tensor_info = tensors_dict[tensor_name]  ## src_node_name, tensor_size

            src_node_name = tensor_info[0]
            src_node_id = NodeId(src_node_name)

            poss_dst_nodes = all_dests_per_tensor[tensor_name]

            net_edge_id = tensor_ass_key.net_edge_id

            if tensor_ass_var.varValue >= INT_VARS_VALUE_THR:
                ## There is at least one node on the receiver server
                ## We have to find these nodes

                actual_dst_nodes = assigned_nodes_dict[net_edge_id[1]].intersection(
                    poss_dst_nodes
                )

                for act_dst_node_id in actual_dst_nodes:
                    mod_edge_id = (src_node_id, act_dst_node_id)

                    ## Renaming tensors according to quantization
                    quantization_key = QuantizationKey(
                        mod_edge_id[0], tensor_ass_key.mod_name
                    )
                    tensor_name_list = []
                    if (
                        quantization_key in quantization_vars
                        and quantization_vars[quantization_key].varValue
                        >= INT_VARS_VALUE_THR
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
        print("Done Analyzing tensor assignments...")

        for quant_key, quant_var in filtered_quant_vars.items():
            if quant_var.varValue >= INT_VARS_VALUE_THR:

                mod_node_id = quant_key.mod_node_id
                solved_model_graph.nodes[mod_node_id][SolvedNodeInfo.QUANTIZED] = True

        return solved_model_graph

    @staticmethod
    def prepare_problem(
        model_profile_list: list[ModelProfile],
        network_graph: nx.DiGraph,
        start_server: NodeId,
        opt_params: OptimizationParams,
        server_execution_profile_pool: ServerExecutionProfilePool,
    ):
        model_graphs: list[nx.DiGraph] = [
            model_profile.get_model_graph() for model_profile in model_profile_list
        ]

        problem: pulp.LpProblem = pulp.LpProblem("Partitioning", pulp.LpMinimize)

        node_ass_vars: dict[NodeAssKey, pulp.LpVariable] = {}
        tensor_ass_vars: dict[TensorAssKey, pulp.LpVariable] = {}
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

            curr_tensor_ass_vars: dict[TensorAssKey, pulp.LpVariable] = (
                VarsBuilder.define_tensor_assignment_vars(curr_mod_graph, network_graph)
            )
            tensor_ass_vars.update(curr_tensor_ass_vars)
            print("Done Defining Tensor Ass Vars")

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
            ConstraintsBuilder.add_tensor_assignment_constraints(
                problem, curr_mod_graph, network_graph, node_ass_vars, tensor_ass_vars
            )

            # ConstraintsBuilder.add_input_flow_constraints(
            #     problem, curr_mod_graph, network_graph, node_ass_vars, tensor_ass_vars
            # )
            # ConstraintsBuilder.add_output_flow_constraints(
            #     problem, curr_mod_graph, network_graph, node_ass_vars, tensor_ass_vars
            # )

            ConstraintsBuilder.add_tensor_ass_vars_product_constraint(
                problem, curr_mod_graph, tensor_ass_vars, quantization_vars
            )

        ConstraintsBuilder.add_node_ass_vars_product_constraint(
            problem, node_ass_vars, quantization_vars
        )

        ConstraintsBuilder.add_memory_constraints(
            problem,
            model_graphs,
            network_graph,
            node_ass_vars,
            mem_use_vars,
        )

        device_energy_expr = EnergyComputer.compute_energy_cost_per_net_node(
            model_graphs,
            network_graph,
            node_ass_vars,
            tensor_ass_vars,
            opt_params.requests_number,
            start_server,
            server_execution_profile_pool,
        )
        if opt_params.device_max_energy > 0:
            ## If <= 0 we assume no boundary
            problem += device_energy_expr <= opt_params.device_max_energy

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
            tensor_ass_vars,
            quantization_vars,
            device_energy_expr,
            regressor_expressions,
        )
