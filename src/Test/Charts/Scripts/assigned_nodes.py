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


def plot_assigned_nodes_for_plan_on_axes(curr_plan: dict):

    whole_assigned_sizes = {}
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
            dev_cpus = result["device_cpus"]
            edge_cpus = result["edge_cpus"]
            cloud_cpus = result["cloud_cpus"]

        curr_assigned_nodes = {}
        for component_id in model_plan.get_all_components():
            component_info = model_plan.plan_dict[component_id]

            server_id = component_id.net_node_id.node_name
            curr_assigned_nodes.setdefault(server_id, [0, 0])

            curr_assigned_nodes[server_id][0] += component_info["component_size"]
            curr_assigned_nodes[server_id][1] += component_info["quantized_nodes_num"]

        whole_assigned_sizes.setdefault(
            (latency_weight, max_noise, dev_cpus, edge_cpus, cloud_cpus),
            curr_assigned_nodes,
        )

    # Prepare list of rows
    rows = []
    for (
        latency_weight,
        max_noises,
        dev_cpus,
        edge_cpus,
        cloud_cpus,
    ), servers in whole_assigned_sizes.items():
        for server_id, layers in servers.items():
            # total_layers = sum(layers)  # if you want total layers per server
            rows.append(
                {
                    "latency_weight": latency_weight,
                    "max_noises": max_noises,
                    "device_cpus": dev_cpus,
                    "edge_cpus": edge_cpus,
                    "cloud_cpus": cloud_cpus,
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

    for dev_cpus in df["device_cpus"].unique():
        for edge_cpus in df["edge_cpus"].unique():
            for cloud_cpus in df["cloud_cpus"].unique():

                df_filter = (
                    (df["device_cpus"] == dev_cpus)
                    & (df["edge_cpus"] == edge_cpus)
                    & (df["cloud_cpus"] == cloud_cpus)
                )

                fig: plt.Figure
                axes: list[list[plt.Axes]]
                fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 7))
                for idx, lw in enumerate(
                    df["latency_weight"].unique()
                ):  # for each latency weightdf["latency_weight"].unique():
                    curr_ax = axes[0][idx]
                    curr_df = df[(df["latency_weight"] == lw) & df_filter]

                    if curr_df.empty:
                        continue

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
                        f"Total Layers - Latency Weight: {lw}; Energy Weight: {1- lw}",
                        fontsize=11,
                    )
                    curr_ax.set_xlabel("Max Noises")
                    curr_ax.set_ylabel("# Assigned Layers")

                for idx, lw in enumerate(
                    df["latency_weight"].unique()
                ):  # for each latency weightdf["latency_weight"].unique():

                    curr_ax = axes[1][idx]
                    curr_df = df[(df["latency_weight"] == lw) & df_filter]
                    if curr_df.empty:
                        continue

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

                if cloud_cpus == -1.0:
                    fig.suptitle(f"Device {dev_cpus} + Edge {edge_cpus} - yolo11x-seg")
                    plt.tight_layout()
                    plt.savefig(
                        f"../Images/Assigned_Nodes/device_edge_assigned_nodes_{dev_cpus}_{edge_cpus}.png"
                    )
                else:
                    fig.suptitle(
                        f"Device {dev_cpus} + Edge {edge_cpus} + Cloud - yolo11x-seg"
                    )
                    plt.tight_layout()
                    plt.savefig(
                        f"../Images/Assigned_Nodes/device_edge_cloud_assigned_nodes_{dev_cpus}_{edge_cpus}_{cloud_cpus}.png"
                    )

    pass


def plot_assigned_components_for_plan_on_axes(curr_plan):
    whole_assigned_comps = {}
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
            dev_cpus = result["device_cpus"]
            edge_cpus = result["edge_cpus"]
            cloud_cpus = result["cloud_cpus"]

        curr_assigned_comps = {}
        for component_id in model_plan.get_all_components():

            server_id = component_id.net_node_id.node_name
            curr_assigned_comps.setdefault(server_id, 0)

            curr_assigned_comps[server_id] += 1

        whole_assigned_comps.setdefault(
            (latency_weight, max_noise, dev_cpus, edge_cpus, cloud_cpus),
            curr_assigned_comps,
        )

    # Prepare list of rows
    rows = []
    for (
        latency_weight,
        max_noises,
        dev_cpus,
        edge_cpus,
        cloud_cpus,
    ), servers in whole_assigned_comps.items():
        for server_id, comps_num in servers.items():
            # total_layers = sum(layers)  # if you want total layers per server
            rows.append(
                {
                    "latency_weight": latency_weight,
                    "max_noises": max_noises,
                    "device_cpus": dev_cpus,
                    "edge_cpus": edge_cpus,
                    "cloud_cpus": cloud_cpus,
                    "server_id": server_id,
                    "num_comps": comps_num,
                }
            )

    # Create DataFrame
    df = pd.DataFrame(rows)
    df["server_id"] = df["server_id"].map({"0": "Device", "1": "Edge", "2": "Cloud"})
    df["Server"] = df["server_id"]

    for dev_cpus in df["device_cpus"].unique():
        for edge_cpus in df["edge_cpus"].unique():
            for cloud_cpus in df["cloud_cpus"].unique():

                df_filter = (
                    (df["device_cpus"] == dev_cpus)
                    & (df["edge_cpus"] == edge_cpus)
                    & (df["cloud_cpus"] == cloud_cpus)
                )

                fig: plt.Figure
                axes: list[plt.Axes]
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
                for idx, lw in enumerate(
                    df["latency_weight"].unique()
                ):  # for each latency weightdf["latency_weight"].unique():
                    curr_ax = axes[idx]
                    curr_df = df[(df["latency_weight"] == lw) & df_filter]

                    if curr_df.empty:
                        continue

                    sns.barplot(
                        data=curr_df,
                        x="max_noises",
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
                    curr_ax.set_ylabel("# Components")

                if cloud_cpus == -1.0:
                    fig.suptitle(f"Device {dev_cpus} + Edge {edge_cpus} - yolo11x-seg")
                    plt.tight_layout()
                    plt.savefig(
                        f"../Images/Assigned_Nodes/device_edge_assigned_comps_{dev_cpus}_{edge_cpus}.png"
                    )
                else:
                    fig.suptitle(
                        f"Device {dev_cpus} + Edge {edge_cpus} + Cloud - yolo11x-seg"
                    )
                    plt.tight_layout()
                    plt.savefig(
                        f"../Images/Assigned_Nodes/device_edge_cloud_assigned_comps_{dev_cpus}_{edge_cpus}_{cloud_cpus}.png"
                    )


def main():

    device_edge_plans = json.loads(open(DEVICE_EDGE_PLANS_PATH).read())
    plot_assigned_nodes_for_plan_on_axes(device_edge_plans)

    device_edge_cloud_plans = json.loads(open(DEVICE_EDGE_CLOUD_PLANS_PATH).read())
    plot_assigned_nodes_for_plan_on_axes(device_edge_cloud_plans)

    device_edge_cloud_plans = json.loads(open(DEVICE_EDGE_CLOUD_PLANS_PATH).read())
    plot_assigned_components_for_plan_on_axes(device_edge_cloud_plans)


if __name__ == "__main__":
    main()
