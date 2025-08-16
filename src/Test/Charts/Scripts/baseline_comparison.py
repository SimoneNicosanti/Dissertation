import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEVICE_USAGE_PATH = "../../Results/DevicePlan/usage.csv"
DEVICE_PLAN_PATH = "../../Results/DevicePlan/generation.csv"

DEVICE_EDGE_USAGE_PATH = "../../Results/DeviceEdgePlan/usage.csv"
DEVICE_EDGE_PLAN_PATH = "../../Results/DeviceEdgePlan/generation.csv"

DEVICE_EDGE_CLOUD_USAGE_PATH = "../../Results/DeviceEdgeCloudPlan/usage.csv"
DEVICE_EDGE_CLOUD_PLAN_PATH = "../../Results/DeviceEdgeCloudPlan/generation.csv"


def main():

    group_cols = [
        "model_name",
        "latency_weight",
        "energy_weight",
        "device_max_energy",
        "max_noises",
        "device_cpus",
        "edge_cpus",
        "cloud_cpus",
        "device_bandwidth",
        "edge_bandwidth",
        "cloud_bandwidth",
    ]

    device_usage_df = (
        pd.read_csv(DEVICE_USAGE_PATH)
        .fillna("None")
        .groupby(group_cols)
        .mean()
        .reset_index()
    )
    device_edge_usage_df = (
        pd.read_csv(DEVICE_EDGE_USAGE_PATH)
        .fillna("None")
        .groupby(group_cols)
        .mean()
        .reset_index()
    )
    device_edge_cloud_usage_df = (
        pd.read_csv(DEVICE_EDGE_CLOUD_USAGE_PATH)
        .fillna("None")
        .groupby(group_cols)
        .mean()
        .reset_index()
    )

    device_plan_gen_df = (
        pd.read_csv(DEVICE_PLAN_PATH)
        .fillna("None")[group_cols + ["energy_value"]]
        .groupby(group_cols)
        .mean()
        .reset_index()
    )
    device_edge_plan_gen_df = (
        pd.read_csv(DEVICE_EDGE_PLAN_PATH)
        .fillna("None")[group_cols + ["energy_value"]]
        .groupby(group_cols)
        .mean()
        .reset_index()
    )
    device_edge_cloud_plan_gen_df = (
        pd.read_csv(DEVICE_EDGE_CLOUD_PLAN_PATH)
        .fillna("None")[group_cols + ["energy_value"]]
        .groupby(group_cols)
        .mean()
        .reset_index()
    )

    models = ["yolo11x-seg"]
    for model in models:
        fig: plt.Figure
        axes: list[list[plt.Axes]]
        fig, axes = plt.subplots(
            figsize=(14, 7),
            nrows=2,
            ncols=len(device_edge_usage_df["latency_weight"].unique()),
            sharey="row",
        )

        latency_weights = device_edge_usage_df["latency_weight"].unique()
        latency_weights = sorted(latency_weights)

        for i, lw in enumerate(latency_weights):
            device_usage_df_i = device_usage_df[device_usage_df["model_name"] == model]
            device_edge_usage_df_i = device_edge_usage_df[
                (device_edge_usage_df["latency_weight"] == lw)
                & (device_edge_usage_df["model_name"] == model)
            ]
            device_edge_cloud_usage_df_i = device_edge_cloud_usage_df[
                (device_edge_usage_df["latency_weight"] == lw)
                & (device_edge_usage_df["model_name"] == model)
            ]

            curr_ax = axes[0][i]

            curr_ax.plot(
                device_usage_df_i["max_noises"].unique(),
                device_usage_df_i["run_time"],
                label="device",
                marker="o",
            )
            curr_ax.plot(
                device_edge_usage_df_i["max_noises"].unique(),
                device_edge_usage_df_i["run_time"],
                label="device + edge",
                marker="o",
            )
            curr_ax.plot(
                device_edge_cloud_usage_df_i["max_noises"].unique(),
                device_edge_cloud_usage_df_i["run_time"],
                label="device + edge + cloud",
                marker="o",
            )

            curr_ax.set_title(f"Latency Weight: {lw}", fontsize=11)

            curr_ax.set_ylabel("Time [s]")
            curr_ax.set_xlabel("Max Noise")

            curr_ax.set_xticks(device_edge_usage_df_i["max_noises"].unique())
            curr_ax.set_xticklabels(curr_ax.get_xticklabels(), rotation=45)

            curr_ax.legend()

        energy_weights = np.sort(device_edge_usage_df["energy_weight"].unique())[::-1]

        for i, ew in enumerate(energy_weights):
            device_plan_gen_df_i = device_plan_gen_df[
                (device_plan_gen_df["model_name"] == model)
            ]
            device_edge_plan_gen_df_i = device_edge_plan_gen_df[
                (device_edge_plan_gen_df["energy_weight"] == ew)
                & (device_edge_plan_gen_df["model_name"] == model)
            ]
            device_edge_cloud_plan_gen_df_i = device_edge_cloud_plan_gen_df[
                (device_edge_cloud_plan_gen_df["energy_weight"] == ew)
                & (device_edge_cloud_plan_gen_df["model_name"] == model)
            ]

            curr_ax = axes[1][i]

            curr_ax.plot(
                device_plan_gen_df_i["max_noises"].unique(),
                device_plan_gen_df_i["energy_value"],
                label="device",
                marker="o",
            )
            curr_ax.plot(
                device_edge_plan_gen_df_i["max_noises"].unique(),
                device_edge_plan_gen_df_i["energy_value"],
                label="device + edge",
                marker="o",
            )
            curr_ax.plot(
                device_edge_cloud_plan_gen_df_i["max_noises"].unique(),
                device_edge_cloud_plan_gen_df_i["energy_value"],
                label="device + edge + cloud",
                marker="o",
            )

            curr_ax.set_title(f"Energy Weight: {ew}", fontsize=11)

            curr_ax.set_ylabel("Energy [J]")
            curr_ax.set_xlabel("Max Noise")

            curr_ax.set_xticks(device_edge_plan_gen_df_i["max_noises"].unique())
            curr_ax.set_xticklabels(curr_ax.get_xticklabels(), rotation=45)

            curr_ax.legend()

        for ax in fig.get_axes():
            ax.tick_params(labelleft=True)  # show tick labels
            ax.yaxis.set_tick_params(which="both", labelleft=True)  # for sa

        fig.suptitle(f"Baseline Comparison. Model: {model}", fontsize=13)

        plt.tight_layout()
        plt.savefig(f"../Images/Base_Comparisons/baseline_comparison_{model}.png")

        pass

    pass


if __name__ == "__main__":
    main()
