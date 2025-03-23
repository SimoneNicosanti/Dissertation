import abc
import json

import networkx as nx


from Common import ConfigReader
from CommonPlan.SolvedModelGraph import ComponentId
from CommonProfile.NodeId import NodeId
import grpc

from proto_compiled.common_pb2 import ModelId
from proto_compiled.model_manager_pb2 import PartitionRequest
from proto_compiled.model_manager_pb2_grpc import ModelManagerStub

class ModelDivider():

    def __init__(self) -> None:
        manager_addr = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "addresses", "MODEL_MANAGER_ADDR"
        )
        manager_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
            "ports", "MODEL_MANAGER_PORT"
        )
        self.manager_chann = grpc.insecure_channel(
            "{}:{}".format(manager_addr, manager_port)
        )

    def divide_model(self, solved_graph: nx.DiGraph, deployment_server: NodeId):
        
        manager_stub = ModelManagerStub(self.manager_chann)

        model_id = ModelId(model_name=solved_graph.graph["name"], deployer_id=deployment_server.node_name)

        solved_graph_json = json.dumps(nx.node_link_data(solved_graph), default=self.encode_complex_info)

        partition_request = PartitionRequest(model_id=model_id, solved_graph=solved_graph_json)

        manager_stub.divide_model(partition_request)

        pass

    def encode_complex_info(self, obj):
        if isinstance(obj, NodeId):
            return {"type" : "NodeId", "node_name" : obj.node_name}
        if isinstance(obj, ComponentId):
            return {"type" : "ComponentId", "net_node_id" : obj.net_node_id.node_name, "component_idx" : obj.component_idx}
        
        raise TypeError("Object {} not serializable".format(obj))

