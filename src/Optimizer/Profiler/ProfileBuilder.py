import abc
import grpc
import networkx as nx
import json

from CommonProfile.NodeId import NodeId
from Common import ConfigReader
from proto_compiled.common_pb2 import ModelId
from proto_compiled.model_manager_pb2 import ProfileRequest, ProfileResponse
from proto_compiled.model_manager_pb2_grpc import ModelManagerStub
import os

class ProfileBuilder(abc.ABC):

    def __init__(self) -> None:
        profiler_addr = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "addresses", "MODEL_MANAGER_ADDR"
        )
        profiler_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
            "ports", "MODEL_MANAGER_PORT"
        )

        self.profiler_channel = grpc.insecure_channel(
            "{}:{}".format(profiler_addr, profiler_port)
        )

    def profile_model(self, model_name : str) -> nx.DiGraph:
        ## Call ModelManager to actually profile model
        print("Profiling Model")
        json_profile = self.read_profile(model_name)
        if json_profile is None :
            model_id = ModelId(model_name=model_name, deployer_id="-1")
            profile_request = ProfileRequest(model_id=model_id)
            print("Sending Profile Request")
            profile_response : ProfileResponse = ModelManagerStub(self.profiler_channel).profile_model(profile_request)
            
            json_profile = profile_response.model_profile
            self.save_profile(model_name, json_profile)
            print("Done Profiling Model")
        else :
            print("Model Already Profiled")

        model_graph: nx.MultiDiGraph = nx.node_link_graph(
            json.loads(json_profile)
        )
        print("Done Loading Graph")

        label_mapping = {
            node_name: NodeId(node_name) for node_name in model_graph.nodes
        }
        model_graph = nx.relabel_nodes(model_graph, label_mapping)
        
        return model_graph



    def save_profile(self, model_name : str,
        profile_json: str
    ):
        profiles_dir = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "optimizer_dirs", "PROFILES_DIR"
        )
        model_profile_path = os.path.join(profiles_dir, model_name + ".json")
        print("Saving Profile")
        with open(model_profile_path, "w") as json_file:
            json_file.write(profile_json)
        pass


    def read_profile(self, model_name: str) -> str:
        profiles_dir = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "optimizer_dirs", "PROFILES_DIR"
        )

        model_profile_path = os.path.join(profiles_dir, model_name + ".json")
        if os.path.isfile(model_profile_path):
            with open(model_profile_path, "r") as json_file:
                return json_file.read()
        return None
