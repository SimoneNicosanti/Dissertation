import json

import grpc

from Common.ConfigReader import ConfigReader
from CommonProfile.NetworkProfile import NetworkProfile
from proto_compiled.common_pb2 import Empty
from proto_compiled.register_pb2 import StateMap
from proto_compiled.register_pb2_grpc import RegisterStub


class NetworkProfileBuilder:
    def __init__(self):

        ## Connect to InfluxDB
        ## Retrieve Network Info
        ## Connect to find static server info (energy consumptions)
        registry_addr = ConfigReader().read_str("addresses", "REGISTRY_ADDR")
        registry_port = ConfigReader().read_int("ports", "REGISTRY_PORT")
        self.registry_chan = grpc.insecure_channel(
            "{}:{}".format(registry_addr, registry_port)
        )

        pass

    def build_network_profile(self) -> NetworkProfile:
        registry_stub: RegisterStub = RegisterStub(self.registry_chan)

        state_map: StateMap = registry_stub.pull_all_states(Empty())
        network_profile_str = state_map.network_profile
        network_profile_dict = json.loads(network_profile_str)
        network_profile = NetworkProfile.decode(network_profile_dict)

        return network_profile
