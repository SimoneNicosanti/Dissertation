import json

import networkx as nx

from CommonIds.NodeId import NodeId
from CommonPlan.Plan import Plan
from CommonPlan.SolvedModelGraph import SolvedGraphInfo
from CommonPlan.WholePlan import WholePlan
from CommonProfile.ExecutionProfile import ServerExecutionProfilePool
from CommonProfile.ModelProfile import ModelProfile
from CommonProfile.NetworkProfile import NetworkProfile
from Optimizer.Builder.PlanBuilder import PlanBuilder
from Optimizer.Optimization.OptimizationHandler import (
    OptimizationHandler,
    OptimizationParams,
)
from Optimizer.PostProcessing.PostProcessor import PostProcessor
from proto_compiled.optimizer_pb2 import OptimizationRequest, OptimizationResponse
from proto_compiled.optimizer_pb2_grpc import OptimizationServicer


class OptmizationServer(OptimizationServicer):

    def __init__(self):
        pass

    def serve_optimization(self, opt_request: OptimizationRequest, context):
        print("Received Optimization Request")

        models_profile_list: list[ModelProfile] = []
        for encoded_model_profile in opt_request.models_profiles:
            model_profile = ModelProfile.decode(json.loads(encoded_model_profile))
            models_profile_list.append(model_profile)
        print("Decoded Model Profiles")

        network_profile: NetworkProfile = NetworkProfile.decode(
            json.loads(opt_request.network_profile)
        )
        print("Decoded Network Profile")

        ## TODO Integrate the execution profile!!
        execution_profile_pool: ServerExecutionProfilePool = (
            ServerExecutionProfilePool.decode(
                json.loads(opt_request.execution_profile_pool)
            )
        )
        print("Decoded Execution Profile Pool")

        model_names = [
            model_profile.get_model_name() for model_profile in models_profile_list
        ]

        optimization_params = OptimizationParams(
            latency_weight=opt_request.latency_weight,
            energy_weight=opt_request.energy_weight,
            device_max_energy=opt_request.device_max_energy,
            requests_number=dict(zip(model_names, opt_request.requests_number)),
            max_noises=dict(zip(model_names, opt_request.max_noises)),
        )

        start_server = NodeId(opt_request.start_server)

        solved_graphs: list[nx.DiGraph] = OptimizationHandler.optimize(
            models_profile_list,
            network_profile.get_network_graph(),
            start_server,
            opt_params=optimization_params,
            server_execution_profile_pool=execution_profile_pool,
        )

        if solved_graphs is None:
            print("No Solution Found")
            return OptimizationResponse(optimized_plan="")

        ## Problem Post Processing
        whole_plan = WholePlan(start_server)
        for solved_graph in solved_graphs:
            graph_name = solved_graph.graph["name"]
            if not solved_graph.graph[SolvedGraphInfo.SOLVED]:
                print(graph_name + " is not solved")
                continue

            PostProcessor.post_process_solution_graph(solved_graph)

            plan: Plan = PlanBuilder.build_plan(solved_graph)

            whole_plan.put_model_plan(graph_name, plan)

        encoded_whole_plan = whole_plan.encode()

        whole_plan_json = json.dumps(encoded_whole_plan)

        return OptimizationResponse(optimized_plan=whole_plan_json)
