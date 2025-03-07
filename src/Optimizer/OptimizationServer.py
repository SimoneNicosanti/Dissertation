

import pickle
from Optimization.OptimizationHandler import OptimizationHandler, OptimizationParams
from Graph.ModelGraph import ModelGraph
from Graph.SolvedModelGraph import SolvedModelGraph
from Graph import ConnectedComponents
from Partitioner.OnnxModelPartitioner import OnnxModelPartitioner
from Profiler.OnnxModelProfiler import OnnxModelProfiler
from proto.optimizer_pb2_grpc import OptimizationServicer
from proto.optimizer_pb2 import OptimizationRequest, OptimizationResponse
import os
import Test



MODEL_PROFILES_DIR = "/optimizer_data/model_profiles/"
MODEL_DIR = "/optimizer_data/models/"

class OptmizationServer(OptimizationServicer):

    def serve_optimization(self, request : OptimizationRequest, context):
        
        optimization_params = OptimizationParams(
            latency_weight = request.latency_weight,
            energy_weight = request.energy_weight,
            device_max_energy = request.device_max_energy,
            requests_number = dict(zip(request.model_names, request.requests_number)))

        model_graphs_dict : dict[str, ModelGraph]= {}

        for model_name in request.model_names:
            model_profile_path = os.path.join(MODEL_PROFILES_DIR, model_name + ".pickle")
            if os.path.isfile(model_profile_path):
                with open(model_profile_path, "rb") as pickle_file:
                    model_graphs_dict[model_name] = pickle.load(pickle_file)
            else :
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

        for solved_graph in solved_graphs:
            graph_name = solved_graph.get_graph_name()
            ConnectedComponents.ConnectedComponentsFinder.find_connected_components(
                solved_graph
            )

            print(
                f"################################## Partitioning {graph_name} ##################################"
            )
            partitioner = OnnxModelPartitioner("./models/" + graph_name + ".onnx")
            partitioner.partition_model(solved_graph)
            print(
                f"###############################################################################################"
            )
        
        
        return OptimizationResponse()
    