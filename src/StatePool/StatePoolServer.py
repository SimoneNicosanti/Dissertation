from readerwriterlock import rwlock

from proto_compiled.common_pb2 import Empty
from proto_compiled.state_pool_pb2 import ServerState, StateMap
from proto_compiled.state_pool_pb2_grpc import StatePoolServicer


class StatePoolServer(StatePoolServicer):

    def __init__(self):
        super().__init__()

        self.state_dict_lock = rwlock.RWLockWriteD()
        self.state_dict: dict[str, str] = {}

    def push_state(self, push_req: ServerState, context):
        print(push_req)
        with self.state_dict_lock.gen_wlock():
            self.state_dict[push_req.server_id] = push_req.state

        return Empty()

    def pull_all_states(self, pull_req: Empty, context):
        with self.state_dict_lock.gen_rlock():
            states = {
                server_id: server_state
                for server_id, server_state in self.state_dict.items()
            }

        return StateMap(state_map=states)
