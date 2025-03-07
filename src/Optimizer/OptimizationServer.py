import os
import pickle

import Test
from Graph import ConnectedComponents
from Graph.ModelGraph import ModelGraph
from Graph.SolvedModelGraph import SolvedModelGraph
from Optimization.OptimizationHandler import OptimizationHandler, OptimizationParams
from Partitioner.OnnxModelPartitioner import OnnxModelPartitioner
from Plan.ModelDistributor import ModelDistributor
from Plan.Plan import Plan
from Plan.PlanDistributor import PlanDistributor
from Profiler.OnnxModelProfiler import OnnxModelProfiler
from proto.common_pb2 import OptimizedPlan
from proto.optimizer_pb2 import OptimizationRequest
from proto.optimizer_pb2_grpc import OptimizationServicer

MODEL_PROFILES_DIR = "/optimizer_data/model_profiles/"
MODEL_DIR = "/optimizer_data/models/"
DIVIDED_MODEL_DIR = "/optimizer_data/divided_models/"


class OptmizationServer(OptimizationServicer):

    def __init__(self):
        self.plan_distributor = PlanDistributor()
        self.model_distributor = ModelDistributor(DIVIDED_MODEL_DIR)
        self.network_builder = None
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
            model_profile_path = os.path.join(
                MODEL_PROFILES_DIR, model_name + ".pickle"
            )
            if os.path.isfile(model_profile_path):
                with open(model_profile_path, "rb") as pickle_file:
                    model_graphs_dict[model_name] = pickle.load(pickle_file)
            else:
                model_path = os.path.join(MODEL_DIR, model_name + ".onnx")
                model_graph: ModelGraph = OnnxModelProfiler(model_path).profile_model(
                    {"args_0": (1, 3, 448, 448)}
                )

                model_graphs_dict[model_name] = model_graph
                with open(model_profile_path, "wb") as pickle_file:
                    pickle.dump(model_graph, pickle_file)

        network_graph = Test.prepare_network_profile(request.deployment_server)
        deployment_server = network_graph.build_node_id(request.deployment_server)

        solved_graphs: list[SolvedModelGraph] = OptimizationHandler().optimize(
            list(model_graphs_dict.values()),
            network_graph,
            deployment_server,
            opt_params=optimization_params,
        )

        plan_map = {}
        for solved_graph in solved_graphs:
            graph_name = solved_graph.get_graph_name()
            ConnectedComponents.ConnectedComponentsFinder.find_connected_components(
                solved_graph
            )

            plan = Plan(solved_graph)

            partitioner = OnnxModelPartitioner(
                MODEL_DIR + graph_name + ".onnx", DIVIDED_MODEL_DIR
            )
            partitioner.partition_model(plan, solved_graph.get_graph_name(), True)

            plan_map[graph_name] = plan.dump_plan()

            self.model_distributor.distribute(graph_name, plan)

        self.plan_distributor.distribute_plan(plan_map, network_graph)

        return OptimizedPlan(
            plans_map=plan_map,
        )
