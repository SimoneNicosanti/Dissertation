import os

import networkx as nx

from Common import ConfigReader
from Optimizer.Divider import ModelDivider
from Optimizer.Graph import ConnectedComponents
from CommonProfile.NodeId import NodeId
from Optimizer.Graph.SolvedModelGraph import SolvedGraphInfo
from Optimizer.Network.NetworkBuilder import NetworkBuilder
from Optimizer.Optimization.OptimizationHandler import (
    OptimizationHandler,
    OptimizationParams,
)
from Optimizer.Divider.ModelDivider import ModelDivider
from Optimizer.Plan.Plan import Plan
from Optimizer.Plan.PlanDistributor import PlanDistributor
from Optimizer.Profiler.ProfileBuilder import ProfileBuilder
from proto_compiled.common_pb2 import OptimizedPlan
from proto_compiled.optimizer_pb2 import OptimizationRequest
from proto_compiled.optimizer_pb2_grpc import OptimizationServicer


class OptmizationServer(OptimizationServicer):

    def __init__(self):
        self.plan_distributor = PlanDistributor()
        self.network_builder = NetworkBuilder()
        self.model_profiler = ProfileBuilder()
        self.model_divider = ModelDivider(None)
        pass

    def serve_optimization(self, request: OptimizationRequest, context):
        print("Received Optimization Request")
        optimization_params = OptimizationParams(
            latency_weight=request.latency_weight,
            energy_weight=request.energy_weight,
            device_max_energy=request.device_max_energy,
            requests_number=dict(zip(request.model_names, request.requests_number)),
        )

        model_graphs_dict: dict[str, nx.MultiDiGraph] = {}
        print("Here")
        for model_name in request.model_names:
            print("Profiling Model ", model_name)
            model_graphs_dict[model_name] = self.model_profiler.profile_model(model_name)
        print("Built Model Graphs and Profiles")

        network_graph: nx.DiGraph = self.network_builder.build_network()
        print("Built Network Graph")
        deployment_server = NodeId(request.deployment_server)

        solved_graphs: list[nx.DiGraph] = OptimizationHandler().optimize(
            list(model_graphs_dict.values()),
            network_graph,
            deployment_server,
            opt_params=optimization_params,
        )
        print("Problem Solved!")

        plan_map = {}
        for solved_graph in solved_graphs:
            graph_name = solved_graph.graph["name"]
            if not solved_graph.graph[SolvedGraphInfo.SOLVED]:
                print(graph_name + " is not solved")
                continue

            ConnectedComponents.ConnectedComponentsFinder.find_connected_components(
                solved_graph
            )

            plan = Plan(solved_graph, deployer_id=deployment_server.node_name)

        #     self.model_divider.divide_model(plan, solved_graph, deployment_server)
        #     print("Partitions Done")

        #     plan_map[graph_name] = plan.dump_plan()

        # print("All Models Parts Distributed")

        # self.plan_distributor.distribute_plan(
        #     plan_map, network_graph, deployment_server.node_name
        # )
        # print("Plan Distributed to Servers")

        # self.write_whole_plan(plan_map, deployment_server.node_name)

        return OptimizedPlan(
            plans_map=plan_map,
        )

    def write_whole_plan(self, plan_map: dict, deployer_id: str):
        plans_dir = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "optimizer_dirs", "PLANS_DIR"
        )
        for model_name, plan_string in plan_map.items():
            plan_name = "plan_depl_{}_{}.json".format(deployer_id, model_name)
            plan_path = os.path.join(plans_dir, plan_name)
            with open(plan_path, "w") as plan_file:
                plan_file.write(plan_string)
