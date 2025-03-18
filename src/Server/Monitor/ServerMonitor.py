import json
import threading
import time

import grpc
from readerwriterlock import rwlock

from Common import ConfigReader
from proto_compiled.common_pb2 import Empty
from proto_compiled.ping_pb2 import PingMessage
from proto_compiled.ping_pb2_grpc import PingStub
from proto_compiled.register_pb2 import AllServerInfo
from proto_compiled.register_pb2_grpc import RegisterStub
from proto_compiled.state_pool_pb2 import ServerState
from proto_compiled.state_pool_pb2_grpc import StatePoolStub

MEGABYTE_SIZE = 1024 * 1024


class ServerMonitor:
    def __init__(self, server_id: str):
        state_pool_addr = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "addresses", "STATE_POOL_ADDR"
        )
        state_pool_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
            "ports", "STATE_POOL_PORT"
        )
        self.state_pool_chan = grpc.insecure_channel(
            "{}:{}".format(state_pool_addr, state_pool_port)
        )

        registry_addr = ConfigReader.ConfigReader("./config/config.ini").read_str(
            "addresses", "REGISTRY_ADDR"
        )
        registry_port = ConfigReader.ConfigReader("./config/config.ini").read_int(
            "ports", "REGISTRY_PORT"
        )
        self.registry_chan = grpc.insecure_channel(
            "{}:{}".format(registry_addr, registry_port)
        )

        self.server_chan_dict: dict[str, grpc.Channel] = {}

        self.server_id = server_id

        self.state_lock = rwlock.RWLockWriteD()
        self.current_state: dict[str, float] = {}
        self.bandwidths: dict[str, float] = {}

        pass

    def __update_and_send_state(self):
        self.__update_state()
        self.__send_state()
        monitor_timer = ConfigReader.ConfigReader("./config/config.ini").read_float(
            "monitor", "MONITOR_TIMER_SEC"
        )
        threading.Timer(monitor_timer, self.__update_and_send_state).start()

    def __send_state(self):
        send_state = {}
        with self.state_lock.gen_rlock():
            send_state = {key: value for key, value in self.current_state.items()}
            send_state["bandwidths"] = {
                key: value for key, value in self.bandwidths.items()
            }

        send_state_str = json.dumps(send_state)
        state_pool_stub: StatePoolStub = StatePoolStub(self.state_pool_chan)

        state_pool_stub.push_state(
            ServerState(server_id=self.server_id, state=send_state_str)
        )

    def __update_state(self):
        ## Ping other devices
        ## Collect my state

        self.__update_bandwidth()
        self.__update_flops()
        self.__update_memory()
        self.__update_energy()

    def __update_energy(self):
        if self.server_id == "0":
            energy = 0.5  ## micro-joule / s
        else:
            energy = 1

        with self.state_lock.gen_wlock():
            self.current_state["comp_energy"] = energy
            self.current_state["trans_energy"] = energy

    def __update_flops(self):
        if self.server_id == "0":
            flops = 2.5 * 10**9
        else:
            flops = 5 * 10**9

        with self.state_lock.gen_wlock():
            self.current_state["flops"] = flops

    def __update_memory(self):
        ## Memory in MB
        if self.server_id == "0":
            mem = 100
        else:
            mem = 16 * 1024

        with self.state_lock.gen_wlock():
            self.current_state["memory"] = mem

    def __update_bandwidth(self):
        registry_stub = RegisterStub(self.registry_chan)
        all_server_info: AllServerInfo = registry_stub.get_all_servers_info(Empty())

        for server_info in all_server_info.all_server_info:
            if server_info.server_id.server_id not in self.server_chan_dict.keys():
                server_chan = grpc.insecure_channel(
                    "{}:{}".format(
                        server_info.reachability_info.ip_address,
                        server_info.reachability_info.ping_port,
                    )
                )
                self.server_chan_dict[server_info.server_id.server_id] = server_chan

            server_chan = self.server_chan_dict[server_info.server_id.server_id]

            if self.server_id == "1" and server_info.server_id.server_id == "0":
                bandwidth = 167
            elif self.server_id == "1" and server_info.server_id.server_id == "1":
                bandwidth = 104
            elif self.server_id == "0" and server_info.server_id.server_id == "1":
                bandwidth = 110
            elif self.server_id == "0" and server_info.server_id.server_id == "0":
                bandwidth = 110
            else:
                bandwidth = self.__eval_bandwidth(server_chan)

            with self.state_lock.gen_wlock():
                self.bandwidths[server_info.server_id.server_id] = bandwidth

    def __eval_bandwidth(self, server_chan: grpc.Channel):
        ping_stub = PingStub(server_chan)

        ping_message_size_mb = ConfigReader.ConfigReader(
            "./config/config.ini"
        ).read_float("monitor", "PING_MESSAGE_SIZE_MB")
        ping_message_content = bytearray(
            ping_message_size_mb * MEGABYTE_SIZE
        )  ## Sending 1MB of data

        ping_times = ConfigReader.ConfigReader("./config/config.ini").read_int(
            "monitor", "PING_TIMES"
        )

        start_time = time.perf_counter_ns()
        for _ in range(ping_times):
            ping_stub.ping(PingMessage(ping_bytes=bytes(ping_message_content)))

        total_time = time.perf_counter_ns() - start_time

        avg_rtt = total_time / ping_times  ## Avg round trip time
        avg_go_time = avg_rtt / 2  ## Avg go time

        avg_bandwidth = (ping_message_size_mb * ping_times) / (
            avg_go_time / 1e9
        )  ## Computing bandwidth in MB/s

        return avg_bandwidth

    def init_monitoring(self):
        self.__update_and_send_state()
