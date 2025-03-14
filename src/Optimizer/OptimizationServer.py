import json
import os
import pickle

from Optimizer.Graph import ConnectedComponents
from Optimizer.Graph.ModelGraph import ModelGraph
from Optimizer.Graph.NetworkGraph import NetworkGraph
from Optimizer.Graph.SolvedModelGraph import SolvedModelGraph
from Optimizer.Network.NetworkBuilder import NetworkBuilder
from Optimizer.Optimization.OptimizationHandler import (
    OptimizationHandler,
    OptimizationParams,
)
from Optimizer.Partitioner.OnnxModelPartitioner import OnnxModelPartitioner
from Optimizer.Plan.ModelDistributor import ModelDistributor
from Optimizer.Plan.Plan import Plan
from Optimizer.Plan.PlanDistributor import PlanDistributor
from Optimizer.Profiler.OnnxModelProfiler import OnnxModelProfiler
from proto_compiled.common_pb2 import OptimizedPlan
from proto_compiled.optimizer_pb2 import OptimizationRequest
from proto_compiled.optimizer_pb2_grpc import OptimizationServicer

MODEL_PROFILES_DIR = "/optimizer_data/models_profiles/"
MODEL_DIR = "/optimizer_data/models/"
DIVIDED_MODEL_DIR = "/optimizer_data/divided_models/"


class OptmizationServer(OptimizationServicer):

    def __init__(self):
        self.plan_distributor = PlanDistributor()
        self.model_distributor = ModelDistributor(DIVIDED_MODEL_DIR)
        self.network_builder = NetworkBuilder()
        pass

    def serve_optimization(self, request: OptimizationRequest, context):

        optimization_params = OptimizationParams(
            latency_weight=request.latency_weight,
            energy_weight=request.energy_weight,
            device_max_energy=request.device_max_energy,
            requests_number=dict(zip(request.model_names, request.requests_number)),
        )

        model_graphs_dict: dict[str, ModelGraph] = {}

        for model_name in request.model_names:
            model_graph = self.build_model_graph(model_name)
            model_graphs_dict[model_name] = model_graph

        network_graph: NetworkGraph = self.network_builder.build_network()
        deployment_server = network_graph.build_node_id(request.deployment_server)

        solved_graphs: list[SolvedModelGraph] = OptimizationHandler().optimize(
            list(model_graphs_dict.values()),
            network_graph,
            deployment_server,
            opt_params=optimization_params,
        )

        plan_map = {}
        for solved_graph in solved_graphs:
            if not solved_graph.is_solved():
                print(solved_graph.get_graph_name() + " is not solved")
                continue

            graph_name = solved_graph.get_graph_name()
            ConnectedComponents.ConnectedComponentsFinder.find_connected_components(
                solved_graph
            )

            plan = Plan(solved_graph, deployer_id=deployment_server.node_name)

            partitioner = OnnxModelPartitioner(
                MODEL_DIR + graph_name + ".onnx", DIVIDED_MODEL_DIR
            )
            partitioner.partition_model(
                plan, solved_graph.get_graph_name(), deployment_server
            )

            plan_map[graph_name] = plan.dump_plan()

            self.model_distributor.distribute(
                graph_name, plan, deployment_server.node_name
            )

        self.plan_distributor.distribute_plan(
            plan_map, network_graph, deployment_server.node_name
        )

        self.write_whole_plan(plan_map, deployment_server.node_name)

        return OptimizedPlan(
            plans_map=plan_map,
        )

    def build_model_graph(self, model_name: str) -> ModelGraph:
        model_profile_path = os.path.join(MODEL_PROFILES_DIR, model_name + ".pickle")
        if os.path.isfile(model_profile_path):
            with open(model_profile_path, "rb") as pickle_file:
                model_graph: ModelGraph = pickle.load(pickle_file)
        else:
            model_path = os.path.join(MODEL_DIR, model_name + ".onnx")
            model_graph: ModelGraph = OnnxModelProfiler(model_path).profile_model(
                {"args_0": (1, 3, 448, 448)}
            )
            with open(model_profile_path, "wb") as pickle_file:
                pickle.dump(model_graph, pickle_file)

        return model_graph

    def write_whole_plan(self, plan_map: dict, deployer_id: str):
        for model_name, plan_string in plan_map.items():
            plan_name = "plan_depl_{}_{}.json".format(deployer_id, model_name)
            with open("/optimizer_data/plans/" + plan_name, "w") as plan_file:
                plan_file.write(plan_string)
