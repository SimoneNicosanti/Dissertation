import configparser
import json
import threading
import time

import grpc
import iperf3
import psutil

from Common import ConfigReader
from proto_compiled.common_pb2 import Empty
from proto_compiled.ping_pb2_grpc import PingStub
from proto_compiled.register_pb2 import AllServerInfo, ServerState
from proto_compiled.register_pb2_grpc import RegisterStub

MEGABYTE_SIZE = 1024 * 1024


class ServerMonitor:
    def __init__(self, server_id: str):

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

        self.flops_value = None

        self.current_state: dict[str, float] = {}
        self.bandwidths: dict[str, float] = {}
        self.latencies: dict[str, float] = {}

        pass

    def __update_and_send_state(self):
        print("Update and Send State")
        self.__update_state()
        self.__send_state()
        monitor_timer = ConfigReader.ConfigReader("./config/config.ini").read_float(
            "monitor", "MONITOR_TIMER_SEC"
        )

        threading.Timer(monitor_timer, self.__update_and_send_state).start()

    def __send_state(self):
        send_state = {key: value for key, value in self.current_state.items()}
        send_state["bandwidths"] = {
            key: value for key, value in self.bandwidths.items()
        }
        send_state["latencies"] = {key: value for key, value in self.latencies.items()}

        send_state_str = json.dumps(send_state)
        registry_stub: RegisterStub = RegisterStub(self.registry_chan)

        registry_stub.push_state(
            ServerState(server_id=self.server_id, state=send_state_str)
        )

    def __update_state(self):
        ## Collect latency and bandwidth to the network
        self.__evaluate_bandwidth_and_latency()

        ## Collect memory stats
        self.__evaluate_memory()

        ## Collect energy stats
        self.__evaluate_energy()

    def __evaluate_energy(self):
        if self.server_id == "0":
            section_name = "device"
        elif self.server_id == "1":
            section_name = "edge"
        else:
            section_name = "cloud"

        ## Reading energy from energy config file
        config = configparser.ConfigParser()
        config.read("./config/energy_config.ini")

        energy_values = config[section_name]
        self.current_state["comp_energy_per_sec"] = float(
            energy_values["COMP_ENERGY_PER_SEC"]
        )
        self.current_state["trans_energy_per_sec"] = float(
            energy_values["TRANS_ENERGY_PER_SEC"]
        )
        self.current_state["trans_energy_base"] = float(
            energy_values["TRANS_ENERGY_BASE"]
        )
        self.current_state["self_trans_energy_per_sec"] = float(
            energy_values["SELF_TRANS_ENERGY_PER_SEC"]
        )
        self.current_state["self_trans_energy_base"] = float(
            energy_values["SELF_TRANS_ENERGY_BASE"]
        )

    def __evaluate_memory(self):

        try:
            # if (os.path.isfile("/sys/fs/cgroup/memory.current") and os.path.isfile("/sys/fs/cgroup/memory.max")):
            ## As psutil does not work properly inside a container, we have to read it from the cgroup
            with open("/sys/fs/cgroup/memory.current", "r") as f:
                memory_current = float(f.read())
            with open("/sys/fs/cgroup/memory.max", "r") as f:
                memory_max = float(f.read())
            available_mem = memory_max - memory_current
        except Exception:
            available_mem = psutil.virtual_memory().available

        available_mem = available_mem / MEGABYTE_SIZE  ## MB

        self.current_state["available_memory"] = available_mem

    def __evaluate_bandwidth_and_latency(self):
        registry_stub = RegisterStub(self.registry_chan)
        all_server_info: AllServerInfo = registry_stub.get_all_servers_info(Empty())

        for server_info in all_server_info.all_server_info:

            ## Default everything to zero
            if server_info.server_id.server_id == self.server_id:
                self.bandwidths[server_info.server_id.server_id] = 0
                self.latencies[server_info.server_id.server_id] = 0
                continue

            if server_info.server_id.server_id not in self.server_chan_dict.keys():
                ## Opening channel for the first time to this server
                server_chan = grpc.insecure_channel(
                    "{}:{}".format(
                        server_info.reachability_info.ip_address,
                        server_info.reachability_info.ping_port,
                    )
                )
                self.server_chan_dict[server_info.server_id.server_id] = server_chan

            server_chan = self.server_chan_dict[server_info.server_id.server_id]

            latency = self.__evaluate_latency(server_chan)

            if latency is not None:
                ## Successfully pinged the server
                self.latencies[server_info.server_id.server_id] = latency

                bandwidth = self.__evaluate_bandwidth(
                    server_info.reachability_info.ip_address
                )

                if bandwidth is not None:
                    self.bandwidths[server_info.server_id.server_id] = bandwidth
                else:
                    if server_info.server_id.server_id in self.latencies.keys():
                        ## Assume server not available and remove latency info
                        self.latencies.pop(server_info.server_id.server_id, None)
                    else:
                        ## Keep previous bandwidth
                        pass
            else:
                ## Could not ping the server
                ## Assume server not available and remove both info
                self.latencies.pop(server_info.server_id.server_id, None)
                self.bandwidths.pop(server_info.server_id.server_id, None)

    def __evaluate_latency(self, server_chan: grpc.Channel):
        try:
            ping_stub = PingStub(server_chan)

            ping_times = ConfigReader.ConfigReader("./config/config.ini").read_int(
                "monitor", "PING_TIMES"
            )
            start = time.perf_counter_ns()
            for _ in range(ping_times):
                ping_stub.latency_test(Empty())
            end = time.perf_counter_ns()
            rtt_ns = (end - start) / ping_times

            latency_ns = rtt_ns / 2
            latency = latency_ns * 1e-9
        except Exception:
            latency = None
        return latency

    def __evaluate_bandwidth(self, server_ip_addr: str):
        iperf3_client = iperf3.Client()

        iperf3_client.server_hostname = server_ip_addr

        iperf3_client.port = ConfigReader.ConfigReader("./config/config.ini").read_int(
            "ports", "IPERF3_PORT"
        )

        iperf3_client.protocol = "tcp"

        iperf3_client.duration = ConfigReader.ConfigReader(
            "./config/config.ini"
        ).read_int("monitor", "IPERF3_TEST_DURATION_SEC")

        iperf3_client.bandwidth = (
            ConfigReader.ConfigReader("./config/config.ini").read_int(
                "monitor", "IPERF3_TEST_BANDWIDTH"
            )
            * 10**9
        )

        run_success = False
        while not run_success:
            test_result = iperf3_client.run()

            if test_result.error:
                run_success = False

                error_string: str = test_result.error
                if error_string.find("busy") == -1:
                    ## Other kind of error
                    return None
                else:
                    ## The server is just busy: wait and try again
                    time.sleep(1)

                print(f"Bandwidth Test to {server_ip_addr} >> ", error_string)
            else:
                run_success = True

        return test_result.sent_MB_s

    def init_monitoring(self):
        self.__update_and_send_state()
