import argparse
import json
import os
import time
import warnings

import grpc
import pandas as pd
from networkx.readwrite import json_graph
from Utils import ProfileGenerator

from Common import ConfigReader
from CommonProfile.ModelProfile import ModelProfile
from proto_compiled.optimizer_pb2 import OptimizationRequest, OptimizationResponse
from proto_compiled.optimizer_pb2_grpc import OptimizationStub

warnings.simplefilter("ignore", category=FutureWarning)


RANDOM_ITERATIONS = 5
RAND_MAIN_BRANCH_SIZE_INCREMENT = 50
RAND_OTHER_BRANCH_SIZE_INCREMENT = 15

STATIC_ITERATIONS = 10
STATIC_BRANCH_SIZE_INCREMENT = 150

SKIP_PROB = 0.15

MAX_NET_NODES = 6

CONFIG_RUN_NUMS = 5


def get_optimizer_stub():

    optimizer_addr = ConfigReader.ConfigReader("../../config/config.ini").read_str(
        "addresses", "OPTIMIZER_ADDR"
    )
    optimizer_port = ConfigReader.ConfigReader("../../config/config.ini").read_int(
        "ports", "OPTIMIZER_PORT"
    )

    channel = grpc.insecure_channel(
        target=f"{optimizer_addr}:{optimizer_port}",
        options=[
            ("grpc.max_send_message_length", 1000 * 1024 * 1024),  # 100 MB
            ("grpc.max_receive_message_length", 1000 * 1024 * 1024),  # 100 MB
        ],
    )

    return OptimizationStub(channel=channel)
    pass


def build_current_network_configuration(generated_profile: ModelProfile, max_net_nodes):
    base_flops = 1e9
    flops_step = 1e3
    base_comp_energy = 3
    comp_energy_step = 2
    base_trans_energy = 2
    trans_energy_step = 2
    base_bandwidth = 30
    bandwidth_step = 5  # decremento per "distanza"
    min_bandwidth = 2.5
    base_latency = 0.005
    latency_step = 0.01  # incremento per "distanza"

    flops_dict = {}
    node_avail_mem_dict = {}
    node_comp_energy_dict = {}
    node_trans_energy_dict = {}
    node_bandwidths_dict = {}
    node_latencies_dict = {}

    for net_node_idx in range(max_net_nodes):
        flops_dict[net_node_idx] = base_flops * (flops_step**net_node_idx)

        node_avail_mem_dict[net_node_idx] = 1e9

        node_comp_energy_dict[net_node_idx] = (
            base_comp_energy + comp_energy_step * net_node_idx
        )
        node_trans_energy_dict[net_node_idx] = (
            base_trans_energy + trans_energy_step * net_node_idx
        )

        for net_node_idx in range(max_net_nodes):
            node_bandwidths_dict[net_node_idx] = {}
            node_latencies_dict[net_node_idx] = {}
            for other_node_idx in range(max_net_nodes):
                if net_node_idx == other_node_idx:
                    # stesso nodo: banda infinita e latenza zero
                    node_bandwidths_dict[net_node_idx][other_node_idx] = 0
                    node_latencies_dict[net_node_idx][other_node_idx] = 0
                else:
                    # distanza fra i due indici
                    dist = abs(net_node_idx - other_node_idx)

                    # banda che decresce linearmente
                    bw = max(
                        base_bandwidth - bandwidth_step * (dist - 1),
                        min_bandwidth,
                    )

                    # latenza che cresce linearmente
                    lat = base_latency + latency_step * (dist - 1)

                    node_bandwidths_dict[net_node_idx][other_node_idx] = bw
                    node_latencies_dict[net_node_idx][other_node_idx] = lat

    profile_pool = ProfileGenerator.build_execution_profile(
        generated_profile.get_model_graph(),
        flops_dict,
    )

    network_profile = ProfileGenerator.build_network_profile(
        range(max_net_nodes),
        node_avail_mem=node_avail_mem_dict,
        node_comp_energy_per_sec=node_comp_energy_dict,
        node_trans_energy_per_sec=node_trans_energy_dict,
        node_bandwidths=node_bandwidths_dict,
        node_latencies=node_latencies_dict,
    )

    return profile_pool, network_profile


def random_test(server_num=None):

    optimizer_stub = get_optimizer_stub()

    with open("../Results/Model_Profile/yolo11x-seg_easy.json") as f:
        data = json.load(f)

    base_profile = json_graph.node_link_graph(data["graph"], directed=True)

    curr_branch_sizes = [500, 50, 50, 50, 50]

    if os.path.exists("../Results/ScaleTest/random_scale_test.csv"):
        # os.remove("../Results/ScaleTest/random_scale_test.csv")
        dataframe = pd.read_csv("../Results/ScaleTest/random_scale_test.csv")
    else:
        dataframe = pd.DataFrame(
            columns=[
                "num_nodes",
                "num_tensors",
                "net_nodes",
                "build_time",
                "latency_time",
                "energy_time",
                "whole_time",
                "post_time",
                "total_time",
            ]
        )

    for _ in range(RANDOM_ITERATIONS):

        generated_profile: ModelProfile = ProfileGenerator.build_model_profile(
            base_profile, curr_branch_sizes, SKIP_PROB
        )

        num_nodes = len(generated_profile.get_model_graph().nodes)
        num_tensors = len(generated_profile.get_model_graph().graph["tensor_size_dict"])

        if server_num is None:
            start_server_idx = 1
            end_server_idx = MAX_NET_NODES
        else:
            start_server_idx = server_num
            end_server_idx = server_num

        for max_net_nodes in range(start_server_idx, end_server_idx + 1):

            dataframe = dataframe[
                ~(
                    (dataframe["num_nodes"] == num_nodes)
                    & (dataframe["num_tensors"] == num_tensors)
                    & (dataframe["net_nodes"] == max_net_nodes)
                )
            ]

            print("Testing Configuration >>")
            print(f"\t Branches >> {curr_branch_sizes}")
            print(f"\t Server Nodes >> {max_net_nodes}")

            profile_pool, network_profile = build_current_network_configuration(
                generated_profile, max_net_nodes
            )

            for _ in range(CONFIG_RUN_NUMS):

                optimization_req = OptimizationRequest(
                    models_profiles=[json.dumps(generated_profile.encode())],
                    network_profile=json.dumps(network_profile.encode()),
                    execution_profile_pool=json.dumps(profile_pool.encode()),
                    latency_weight=0.5,
                    energy_weight=0.5,
                    device_max_energy=0,
                    requests_number=[1],
                    max_noises=[0.0],
                    start_server="0",
                )

                start = time.perf_counter_ns()
                optimization_response: OptimizationResponse = (
                    optimizer_stub.serve_optimization(optimization_req)
                )
                end = time.perf_counter_ns()

                plan_time = (end - start) * 1e-9

                dataframe.loc[len(dataframe)] = [
                    num_nodes,
                    num_tensors,
                    max_net_nodes,
                    optimization_response.problem_build_time,
                    optimization_response.min_latency_sol_time,
                    optimization_response.min_energy_sol_time,
                    optimization_response.whole_sol_time,
                    optimization_response.post_processing_time,
                    plan_time,
                ]

                dataframe.to_csv(
                    "../Results/ScaleTest/random_scale_test.csv", index=False
                )

        for i in range(len(curr_branch_sizes)):
            if i == 0:
                increment = RAND_MAIN_BRANCH_SIZE_INCREMENT
            else:
                increment = RAND_OTHER_BRANCH_SIZE_INCREMENT
            curr_branch_sizes[i] += increment

        pass


def static_test(server_num=None):

    optimizer_stub = get_optimizer_stub()

    curr_branch_sizes = [500]

    if os.path.exists("../Results/ScaleTest/static_scale_test.csv"):
        dataframe = pd.read_csv("../Results/ScaleTest/static_scale_test.csv")
    else:
        dataframe = pd.DataFrame(
            columns=[
                "num_nodes",
                "num_tensors",
                "net_nodes",
                "build_time",
                "latency_time",
                "energy_time",
                "whole_time",
                "post_time",
                "total_time",
            ]
        )

    for _ in range(STATIC_ITERATIONS):

        generated_profile: ModelProfile = ProfileGenerator.build_model_profile(
            None, curr_branch_sizes, 0.0, False
        )

        num_nodes = len(generated_profile.get_model_graph().nodes)
        num_tensors = len(generated_profile.get_model_graph().graph["tensor_size_dict"])

        if server_num is None:
            start_server_idx = 1
            end_server_idx = MAX_NET_NODES
        else:
            start_server_idx = server_num
            end_server_idx = server_num

        for max_net_nodes in range(start_server_idx, end_server_idx + 1):

            dataframe = dataframe[
                ~(
                    (dataframe["num_nodes"] == num_nodes)
                    & (dataframe["num_tensors"] == num_tensors)
                    & (dataframe["net_nodes"] == max_net_nodes)
                )
            ]

            print("Testing Configuration >>")
            print(f"\t Branches >> {curr_branch_sizes}")
            print(f"\t Server Nodes >> {max_net_nodes}")

            profile_pool, network_profile = build_current_network_configuration(
                generated_profile, max_net_nodes
            )

            for _ in range(CONFIG_RUN_NUMS):

                optimization_req = OptimizationRequest(
                    models_profiles=[json.dumps(generated_profile.encode())],
                    network_profile=json.dumps(network_profile.encode()),
                    execution_profile_pool=json.dumps(profile_pool.encode()),
                    latency_weight=0.5,
                    energy_weight=0.5,
                    device_max_energy=0,
                    requests_number=[1],
                    max_noises=[0.0],
                    start_server="0",
                )

                start = time.perf_counter_ns()
                optimization_response: OptimizationResponse = (
                    optimizer_stub.serve_optimization(optimization_req)
                )
                end = time.perf_counter_ns()

                plan_time = (end - start) * 1e-9

                dataframe.loc[len(dataframe)] = [
                    num_nodes,
                    num_tensors,
                    max_net_nodes,
                    optimization_response.problem_build_time,
                    optimization_response.min_latency_sol_time,
                    optimization_response.min_energy_sol_time,
                    optimization_response.whole_sol_time,
                    optimization_response.post_processing_time,
                    plan_time,
                ]

                dataframe.to_csv(
                    "../Results/ScaleTest/static_scale_test.csv", index=False
                )

        for i in range(len(curr_branch_sizes)):
            if i == 0:
                increment = STATIC_BRANCH_SIZE_INCREMENT
            else:
                increment = STATIC_BRANCH_SIZE_INCREMENT
            curr_branch_sizes[i] += increment

        pass


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=str,
        help="Test Case",
        required=True,
        choices=["static", "random"],
    )
    parser.add_argument("--server-num", type=int, help="Server Number", default=None)

    args = parser.parse_args()

    if args.case == "random":
        random_test(args.server_num)
    else:
        static_test(args.server_num)

    # random_test()


if __name__ == "__main__":
    main()
