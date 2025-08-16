import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src folder to sys.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from CommonPlan.Plan import Plan

# DEVICE_PLANS_PATH = "../../Results/DevicePlan/plan.json"
DEVICE_EDGE_PLANS_PATH = "../../Results/DeviceEdgePlan/plan.json"
DEVICE_EDGE_CLOUD_PLANS_PATH = "../../Results/DeviceEdgeCloudPlan/plan.json"


pattern = re.compile(
    r"^(?P<model_name>[^_]+)"  # model name until first underscore
    r"_lw_(?P<latency_weight>-?\d+(?:\.\d+)?)"
    r"_ew_(?P<energy_weight>-?\d+(?:\.\d+)?)"
    r"_dme_(?P<device_max_energy>-?\d+(?:\.\d+)?)"
    r"_mn_(?P<max_noises>-?\d+(?:\.\d+)?)"
    r"_dc_(?P<device_cpus>-?\d+(?:\.\d+)?)"
    r"_ec_(?P<edge_cpus>-?\d+(?:\.\d+)?)"
    r"_cc_(?P<cloud_cpus>-?\d+(?:\.\d+)?)"
    r"_db_(?P<device_bandwidth>-?\d+(?:\.\d+)?)"
    r"_eb_(?P<edge_bandwidth>-?\d+(?:\.\d+)?)"
    r"_cb_(?P<cloud_bandwidth>-?\d+(?:\.\d+)?)$"
)


def build_plan_string(
    model_name: str,
    latency_weight: float,
    energy_weight: float,
    device_max_energy: float,
    max_noises: float,
    device_cpus: float,
    edge_cpus: float,
    cloud_cpus: float,
    device_bandwidth: float,
    edge_bandwidth: float,
    cloud_bandwidth: float,
) -> str:
    return (
        f"{model_name}"
        f"_lw_{latency_weight}"
        f"_ew_{energy_weight}"
        f"_dme_{device_max_energy}"
        f"_mn_{max_noises}"
        f"_dc_{device_cpus}"
        f"_ec_{edge_cpus}"
        f"_cc_{cloud_cpus}"
        f"_db_{device_bandwidth}"
        f"_eb_{edge_bandwidth}"
        f"_cb_{cloud_bandwidth}"
    )


def plot_assigned_nodes_for_plan_on_axes(
    curr_plan: dict, ax_list: list[list[plt.Axes]]
):

    whole_assigned_sizes = {}
    latency_weights = set()
    max_noises = set()
    for key in curr_plan.keys():

        model_plan = Plan.decode(curr_plan[key])

        match = pattern.match(key)
        if match:
            result = match.groupdict()
            # convert numeric fields to int/float
            for k, v in result.items():
                if k == "model_name":
                    continue
                result[k] = float(v) if "." in v else int(v)

            max_noise = result["max_noises"]
            latency_weight = result["latency_weight"]

            max_noises.add(max_noise)
            latency_weights.add(latency_weight)

        curr_assigned_nodes = {}
        for component_id in model_plan.get_all_components():
            component_info = model_plan.plan_dict[component_id]

            server_id = component_id.net_node_id.node_name
            curr_assigned_nodes.setdefault(server_id, [0, 0])

            curr_assigned_nodes[server_id][0] += component_info["component_size"]
            curr_assigned_nodes[server_id][1] += component_info["quantized_nodes_num"]

        whole_assigned_sizes.setdefault(
            (latency_weight, max_noise), curr_assigned_nodes
        )

    # Prepare list of rows
    rows = []
    for (latency_weight, max_noises), servers in whole_assigned_sizes.items():
        for server_id, layers in servers.items():
            # total_layers = sum(layers)  # if you want total layers per server
            rows.append(
                {
                    "latency_weight": latency_weight,
                    "max_noises": max_noises,
                    "server_id": server_id,
                    "num_layers": layers[0],
                    "quant_layers": layers[1],
                }
            )

    # Create DataFrame
    df = pd.DataFrame(rows)
    df["server_id"] = df["server_id"].map({"0": "Device", "1": "Edge", "2": "Cloud"})
    df["Server"] = df["server_id"]
    df["Num Layers"] = df["num_layers"]
    df["Max Noise"] = df["max_noises"]

    for idx, lw in enumerate(
        df["latency_weight"].unique()
    ):  # for each latency weightdf["latency_weight"].unique():
        curr_ax = ax_list[idx][0]
        curr_df = df[df["latency_weight"] == lw]
        sns.barplot(
            data=curr_df,
            x="Max Noise",
            y="Num Layers",
            hue="Server",
            ax=curr_ax,
            hue_order=["Device", "Edge", "Cloud"],
        )

        max_height = curr_df["num_layers"].max()
        curr_ax.set_ylim(0, max_height * 1.1)  # 10% extra space

        # Annotate bars with their height
        for p in curr_ax.patches:
            # Only annotate bars with positive height
            if p.get_height() > 0 and p.get_y() >= 0:
                curr_ax.text(
                    x=p.get_x() + p.get_width() / 2,
                    y=p.get_height(),
                    s=int(p.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        curr_ax.set_title(
            f"Total Layers - Latency Weight: {lw}; Energy Weight: {1- lw}", fontsize=11
        )
        curr_ax.set_xlabel("Max Noises")
        curr_ax.set_ylabel("# Assigned Layers")

    for idx, lw in enumerate(
        df["latency_weight"].unique()
    ):  # for each latency weightdf["latency_weight"].unique():

        curr_ax = ax_list[idx][1]
        curr_df = df[df["latency_weight"] == lw]
        sns.barplot(
            data=curr_df,
            x="Max Noise",
            y="quant_layers",
            hue="Server",
            ax=curr_ax,
            hue_order=["Device", "Edge", "Cloud"],
        )

        max_height = curr_df["quant_layers"].max()
        curr_ax.set_ylim(0, max_height * 1.1)  # 10% extra space

        # Annotate bars with their height
        for p in curr_ax.patches:
            # Only annotate bars with positive height
            if p.get_height() > 0 and p.get_y() >= 0:
                curr_ax.text(
                    x=p.get_x() + p.get_width() / 2,
                    y=p.get_height(),
                    s=int(p.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        curr_ax.set_title(
            f"Quantized Layers - Latency Weight: {lw}; Energy Weight: {1- lw}",
            fontsize=11,
        )
        curr_ax.set_xlabel("Max Noises")
        curr_ax.set_ylabel("# Quantized Layers")

    pass


def plot_assigned_components_for_plan_on_axes(curr_plan, ax_list):
    whole_assigned_sizes = {}
    latency_weights = set()
    max_noises = set()
    for key in curr_plan.keys():

        model_plan = Plan.decode(curr_plan[key])

        match = pattern.match(key)
        if match:
            result = match.groupdict()
            # convert numeric fields to int/float
            for k, v in result.items():
                if k == "model_name":
                    continue
                result[k] = float(v) if "." in v else int(v)

            max_noise = result["max_noises"]
            latency_weight = result["latency_weight"]

            max_noises.add(max_noise)
            latency_weights.add(latency_weight)

        curr_assigned_comps = {}
        for component_id in model_plan.get_all_components():
            component_info = model_plan.plan_dict[component_id]

            server_id = component_id.net_node_id.node_name
            curr_assigned_comps.setdefault(server_id, 0)

            curr_assigned_comps[server_id] += 1

        whole_assigned_sizes.setdefault(
            (latency_weight, max_noise), curr_assigned_comps
        )

    # Prepare list of rows
    rows = []
    for (latency_weight, max_noises), servers in whole_assigned_sizes.items():
        for server_id, comps in servers.items():
            # total_layers = sum(layers)  # if you want total layers per server
            rows.append(
                {
                    "latency_weight": latency_weight,
                    "max_noises": max_noises,
                    "server_id": server_id,
                    "num_comps": comps,
                }
            )

    # Create DataFrame
    df = pd.DataFrame(rows)
    df["server_id"] = df["server_id"].map({"0": "Device", "1": "Edge", "2": "Cloud"})
    df["Server"] = df["server_id"]
    df["Max Noise"] = df["max_noises"]

    for idx, lw in enumerate(
        df["latency_weight"].unique()
    ):  # for each latency weightdf["latency_weight"].unique():
        curr_ax = ax_list[idx]
        curr_df = df[df["latency_weight"] == lw]
        sns.barplot(
            data=curr_df,
            x="Max Noise",
            y="num_comps",
            hue="Server",
            ax=curr_ax,
            hue_order=["Device", "Edge", "Cloud"],
        )

        max_height = curr_df["num_comps"].max()
        curr_ax.set_ylim(0, max_height * 1.1)  # 10% extra space

        # Annotate bars with their height
        for p in curr_ax.patches:
            # Only annotate bars with positive height
            if p.get_height() > 0 and p.get_y() >= 0:
                curr_ax.text(
                    x=p.get_x() + p.get_width() / 2,
                    y=p.get_height(),
                    s=int(p.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        curr_ax.set_title(
            f"Total Components - Latency Weight: {lw}; Energy Weight: {1- lw}",
            fontsize=11,
        )
        curr_ax.set_xlabel("Max Noises")
        curr_ax.set_ylabel("Components Num")

    pass
    pass


def main():

    # device_plans = json.loads(open(DEVICE_PLANS_PATH).read())
    device_edge_plans = json.loads(open(DEVICE_EDGE_PLANS_PATH).read())
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10))
    plot_assigned_nodes_for_plan_on_axes(device_edge_plans, axes)
    fig.suptitle("Device + Edge - yolo11x-seg")
    plt.tight_layout()
    plt.savefig("../Images/Assigned_Nodes/device_edge_assigned_nodes.png")

    device_edge_cloud_plans = json.loads(open(DEVICE_EDGE_CLOUD_PLANS_PATH).read())
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10))
    plot_assigned_nodes_for_plan_on_axes(device_edge_cloud_plans, axes)
    fig.suptitle("Device + Edge + Cloud - yolo11x-seg")
    plt.tight_layout()
    plt.savefig("../Images/Assigned_Nodes/device_edge_cloud_assigned_nodes.png")

    device_edge_cloud_plans = json.loads(open(DEVICE_EDGE_CLOUD_PLANS_PATH).read())
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10))
    plot_assigned_components_for_plan_on_axes(device_edge_cloud_plans, axes)
    fig.suptitle("Device + Edge + Cloud - yolo11x-seg")
    plt.tight_layout()
    plt.savefig("../Images/Assigned_Nodes/device_edge_cloud_assigned_components.png")


if __name__ == "__main__":
    main()
