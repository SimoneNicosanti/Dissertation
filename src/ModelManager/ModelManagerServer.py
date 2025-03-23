import json
import networkx as nx



from CommonPlan.Plan import Plan
from ModelManager.Divide import PlanBuilder
from ModelManager.Divide.OnnxModelPartitioner import OnnxModelPartitioner
from ModelManager.Profile.OnnxModelProfiler import OnnxModelProfiler
from ModelManager.Profile import ProfileSaver
from proto_compiled.model_manager_pb2 import PartitionRequest, PartitionResponse, ProfileRequest, ProfileResponse
from proto_compiled.model_manager_pb2_grpc import ModelManagerServicer


class ModelManagerServer(ModelManagerServicer) :
    def __init__(self) :
        pass

    def profile_model(self, profile_request : ProfileRequest, context) -> ProfileResponse:
        print("Received Profile Request")
        model_name = profile_request.model_id.model_name
        profile : str = ProfileSaver.read_profile(model_name)
        if profile is None:
            print("Profiling Model")
            model_profiler : OnnxModelProfiler = OnnxModelProfiler(model_name)
            model_graph : nx.DiGraph = model_profiler.profile_model(None)
            ProfileSaver.save_profile(model_graph)

            profile = ProfileSaver.read_profile(model_name)
        
        return ProfileResponse(model_profile=profile)
    
    def divide_model(self, partition_request : PartitionRequest, context):
        print("Received Partition Request")
        model_plan : Plan = PlanBuilder.build_plan(partition_request.solved_graph, partition_request.model_id.model_name, partition_request.model_id.deployer_id)

        model_partitioner = OnnxModelPartitioner()
        model_partitioner.partition_model(model_plan, partition_request.model_id.model_name, partition_request.model_id.deployer_id)

        return PartitionResponse()