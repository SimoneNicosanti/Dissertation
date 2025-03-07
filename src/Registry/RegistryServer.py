
from proto.common_pb2 import Empty
from proto.register_pb2 import (
    AllServerInfo,
    ReachabilityInfo,
    RegisterResponse,
    ServerInfo,
)
from proto.register_pb2_grpc import RegisterServicer


class RegistryServer(RegisterServicer):

    def __init__(
        self,
    ):
        self.reachability_dict: dict[str, ReachabilityInfo] = {}
        self.server_id = 0
        pass

    def register_server(
        self, reachability_info: ReachabilityInfo, context
    ) -> RegisterResponse:

        new_server_id = str(self.server_id)
        self.reachability_dict[new_server_id] = reachability_info

        register_response = RegisterResponse(server_id=new_server_id)
        self.server_id += 1
        self.log_server_registration(reachability_info, new_server_id)

        return register_response

    def get_all_servers_info(self, request: Empty, context) -> AllServerInfo:

        server_info_list: list[ServerInfo] = []
        for server_id, reach_info in self.reachability_dict.items():
            server_info = ServerInfo(reach_info, server_id)
            server_info_list.append(server_info)

        return AllServerInfo(server_info_list)

    def log_server_registration(self, reachability_info: ReachabilityInfo, server_id : int) -> None:
        print("Received Registration")
        print(f"\t IP Addr >> {reachability_info.ip_address}")
        print(f"\t Port >> {reachability_info.assignment_port}")
        print(f"\t Ping Port >> {reachability_info.ping_port}")
        print(f"\t Server ID >> {server_id}")
        print("----------------------------------------")
