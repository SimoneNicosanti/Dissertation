from readerwriterlock import rwlock

from proto_compiled.common_pb2 import Empty
from proto_compiled.register_pb2 import (
    AllServerInfo,
    ReachabilityInfo,
    ServerId,
    ServerInfo,
)
from proto_compiled.register_pb2_grpc import RegisterServicer


class RegistryServer(RegisterServicer):

    def __init__(
        self,
    ):
        self.reach_dict_lock = rwlock.RWLockWriteD()
        self.reachability_dict: dict[str, tuple[ReachabilityInfo, int]] = {}

        self.server_id_lock = rwlock.RWLockWriteD()
        self.server_id = 0

        pass

    def register_server(self, reachability_info: ReachabilityInfo, context) -> ServerId:
        with self.reach_dict_lock.gen_rlock():
            if reachability_info.ip_address in self.reachability_dict:
                print("Server already registered")
                server_id = self.reachability_dict[reachability_info.ip_address][1]
                return ServerId(server_id=str(server_id))

        with self.server_id_lock.gen_wlock():
            new_server_id = str(self.server_id)
            self.server_id += 1

        with self.reach_dict_lock.gen_wlock():
            self.reachability_dict[reachability_info.ip_address] = [
                reachability_info,
                new_server_id,
            ]

        register_response = ServerId(server_id=new_server_id)
        self.log_server_registration(reachability_info, new_server_id)

        return register_response

    def get_all_servers_info(self, request: Empty, context) -> AllServerInfo:

        server_info_list: list[ServerInfo] = []
        with self.reach_dict_lock.gen_rlock():
            for reach_info, server_id in self.reachability_dict.values():
                server_info = ServerInfo(
                    reachability_info=reach_info,
                    server_id=ServerId(server_id=server_id),
                )
                server_info_list.append(server_info)

        return AllServerInfo(all_server_info=server_info_list)

    def get_info_from_id(self, server_id: ServerId, context) -> ReachabilityInfo:
        with self.reach_dict_lock.gen_rlock():
            for reach_info, id in self.reachability_dict.values():
                if str(id) == server_id.server_id:
                    return reach_info

        raise Exception("Server not found")

    def log_server_registration(
        self, reachability_info: ReachabilityInfo, server_id: int
    ) -> None:
        print("Received Registration")
        print(f"\t IP Addr >> {reachability_info.ip_address}")
        print(f"\t Assignee Port >> {reachability_info.assignment_port}")
        print(f"\t Inference Port >> {reachability_info.inference_port}")
        print(f"\t Ping Port >> {reachability_info.ping_port}")
        print(f"\t Server ID >> {server_id}")
        print("----------------------------------------")
