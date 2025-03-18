import os

import networkx as nx

from Optimizer.Graph import ConnectedComponents
from Optimizer.Graph.Graph import NodeId, SolvedGraphInfo, SolvedNodeInfo
from Optimizer.Network.NetworkBuilder import NetworkBuilder
from Optimizer.Optimization.OptimizationHandler import (
    OptimizationHandler,
    OptimizationParams,
)
from Optimizer.Partitioner.OnnxModelPartitioner import OnnxModelPartitioner
from Optimizer.Plan.ModelDistributor import ModelDistributor
from Optimizer.Plan.Plan import Plan
from Optimizer.Plan.PlanDistributor import PlanDistributor
from Optimizer.Profiler import ProfileSaver
from Optimizer.Profiler.OnnxModelProfiler import OnnxModelProfiler
from proto_compiled.common_pb2 import OptimizedPlan
from proto_compiled.optimizer_pb2 import OptimizationRequest
from proto_compiled.optimizer_pb2_grpc import OptimizationServicer

MODEL_DIR = "/optimizer_data/models/"
DIVIDED_MODEL_DIR = "/optimizer_data/divided_models/"


class OptmizationServer(OptimizationServicer):

    def __init__(self):
        self.plan_distributor = PlanDistributor()
        self.model_distributor = ModelDistributor(DIVIDED_MODEL_DIR)
        self.network_builder = NetworkBuilder()
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

        for model_name in request.model_names:
            model_graph = self.build_model_graph(model_name)
            model_graphs_dict[model_name] = model_graph
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

            partitioner = OnnxModelPartitioner(
                MODEL_DIR + graph_name + ".onnx", DIVIDED_MODEL_DIR
            )
            # partitioner.partition_model(plan, graph_name, deployment_server)

            plan_map[graph_name] = plan.dump_plan()

            # self.model_distributor.distribute(
            #     graph_name, plan, deployment_server.node_name
            # )
        print("All Models Parts Distributed")

        # self.plan_distributor.distribute_plan(
        #     plan_map, network_graph, deployment_server.node_name
        # )
        # print("Plan Distributed to Servers")

        self.write_whole_plan(plan_map, deployment_server.node_name)

        return OptimizedPlan(
            plans_map=plan_map,
        )

    def build_model_graph(self, model_name: str) -> nx.MultiDiGraph:
        model_graph = ProfileSaver.read_profile(model_name)

        if model_graph is None:
            model_path = os.path.join(MODEL_DIR, model_name + ".onnx")
            model_graph: nx.MultiDiGraph = OnnxModelProfiler(model_path).profile_model(
                {"args_0": (1, 3, 448, 448)}
            )
            ProfileSaver.save_profile(model_graph)

        return model_graph

    def write_whole_plan(self, plan_map: dict, deployer_id: str):
        for model_name, plan_string in plan_map.items():
            plan_name = "plan_depl_{}_{}.json".format(deployer_id, model_name)
            with open("/optimizer_data/plans/" + plan_name, "w") as plan_file:
                plan_file.write(plan_string)
