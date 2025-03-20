import networkx as nx



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
        return PartitionResponse()