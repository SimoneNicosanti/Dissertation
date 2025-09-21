import json
import warnings

import grpc
import numpy as np
import pandas as pd
from cv2 import completeSymm

from Common import ConfigReader
from CommonPlan.Plan import Plan
from CommonPlan.WholePlan import WholePlan
from CommonProfile.ModelProfile import ModelProfile
from CommonProfile.NetworkProfile import NetworkProfile
from proto_compiled.optimizer_pb2 import OptimizationRequest, OptimizationResponse
from proto_compiled.optimizer_pb2_grpc import OptimizationStub
from Test.Scripts.Utils import ProfileGenerator

warnings.simplefilter("ignore", category=FutureWarning)


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


def build_current_network_configuration(
    model_profile: ModelProfile, max_net_nodes: int
):

    flops_dict = {0: 1e12, 1: 1e11}
    node_avail_mem_dict = {0: 1e12, 1: 1e12}
    node_comp_energy_dict = {0: 5.0, 1: 2.5}
    node_trans_energy_dict = {0: 2.0, 1: 2.0}

    node_bandwidths_dict = {0: {0: 0.0, 1: 20.0}, 1: {0: 20.0, 1: 0.0}}
    node_latencies_dict = {0: {0: 0.0, 1: 0.005}, 1: {0: 0.005, 1: 0.0}}

    profile_pool = ProfileGenerator.build_execution_profile(
        model_profile.get_model_graph(),
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


def extract_plan_info(plan: Plan):

    device_energy = plan.get_device_energy()

    components_counter = {}
    layers_counter = {}
    for component in plan.get_all_components():
        component_server = component.net_node_id.node_name

        component_info = plan.plan_dict[component]
        num_layers = component_info["component_size"]

        components_counter.setdefault(component_server, 0)
        components_counter[component_server] += 1

        layers_counter.setdefault(component_server, 0)
        layers_counter[component_server] += num_layers

    return device_energy, components_counter, layers_counter


def main():
    optimizer_stub = get_optimizer_stub()

    with open("../Results/Model_Profile/yolo11x-seg_easy.json") as f:
        data = json.load(f)

    model_profile = ModelProfile.decode(data)

    profile_pool, network_profile = build_current_network_configuration(
        model_profile, 2
    )

    dataframe = pd.DataFrame(
        columns=[
            "energy_limit",
            "device_energy",
            "device_layers",
            "edge_layers",
            "device_comps",
            "edge_comps",
        ]
    )

    optimization_req = OptimizationRequest(
        models_profiles=[json.dumps(model_profile.encode())],
        network_profile=json.dumps(network_profile.encode()),
        execution_profile_pool=json.dumps(profile_pool.encode()),
        latency_weight=1.0,
        energy_weight=0.0,
        device_max_energy=0.0,
        requests_number=[1],
        max_noises=[0.0],
        start_server="0",
    )

    optimization_response: OptimizationResponse = optimizer_stub.serve_optimization(
        optimization_req
    )
    whole_plan_string = optimization_response.optimized_plan
    whole_plan_dict = json.loads(whole_plan_string)
    whole_plan = WholePlan.decode(whole_plan_dict)
    plan: Plan = whole_plan.get_model_plan(model_profile.get_model_name())
    no_limit_energy, _, _ = extract_plan_info(plan)

    limit_step = 0.05
    for energy_limit in np.arange(0.0, no_limit_energy + limit_step, limit_step):

        optimization_req = OptimizationRequest(
            models_profiles=[json.dumps(model_profile.encode())],
            network_profile=json.dumps(network_profile.encode()),
            execution_profile_pool=json.dumps(profile_pool.encode()),
            latency_weight=1.0,
            energy_weight=0.0,
            device_max_energy=energy_limit,
            requests_number=[1],
            max_noises=[0.0],
            start_server="0",
        )

        optimization_response: OptimizationResponse = optimizer_stub.serve_optimization(
            optimization_req
        )

        whole_plan_string = optimization_response.optimized_plan

        if len(whole_plan_string) == 0:
            print("No Plan Found")
            new_row = [energy_limit, -1, 0, 0, 0, 0]
        else:
            whole_plan_dict = json.loads(whole_plan_string)
            whole_plan = WholePlan.decode(whole_plan_dict)
            plan: Plan = whole_plan.get_model_plan(model_profile.get_model_name())
            device_energy_value, comps_ass_dict, layers_ass_dict = extract_plan_info(
                plan
            )

            new_row = [energy_limit]
            new_row.append(device_energy_value)
            new_row.append(layers_ass_dict.get("0", 0))
            new_row.append(layers_ass_dict.get("1", 0))
            new_row.append(comps_ass_dict.get("0", 0))
            new_row.append(comps_ass_dict.get("1", 0))

        dataframe.loc[len(dataframe)] = new_row

    dataframe.to_csv("../Results/EnergyLimit/energy_limit.csv", index=False)

    pass


if __name__ == "__main__":
    main()
