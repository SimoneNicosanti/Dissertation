import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

GENERATION_PATH = "../../Results/DeviceEdgeCloudPlan/generation.csv"
USAGE_PATH = "../../Results/DeviceEdgeCloudPlan/usage.csv"


def plan_vs_real_comparison():

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

    plan_df = pd.read_csv(GENERATION_PATH)
    plan_df = plan_df[group_cols + ["latency_value", "energy_value"]]

    usage_df = pd.read_csv(USAGE_PATH)
    usage_df = usage_df[group_cols + ["run_time"]]

    models = ["yolo11x-seg"]
    for model in models:
        for dev_cpus in usage_df["device_cpus"].unique():
            for edge_cpus in usage_df["edge_cpus"].unique():
                axes: list[list[plt.Axes]]
                fig: plt.Figure
                fig, axes = plt.subplots(
                    figsize=(14, 7),
                    nrows=2,
                    ncols=len(plan_df["latency_weight"].unique()),
                    sharey="row",
                )

                latency_weights = plan_df["latency_weight"].unique()
                latency_weights.sort()

                for _, (lw, curr_ax) in enumerate(zip(latency_weights, axes[0])):
                    curr_plan_df = plan_df[
                        (plan_df["latency_weight"] == lw)
                        & (plan_df["model_name"] == model)
                        & (plan_df["device_cpus"] == dev_cpus)
                        & (plan_df["edge_cpus"] == edge_cpus)
                    ]
                    curr_ax.plot(
                        curr_plan_df["max_noises"].unique(),
                        curr_plan_df["latency_value"]
                        .groupby(curr_plan_df["max_noises"])
                        .mean(),
                        color="orange",
                        label="Planned Latency",
                        marker="o",
                    )

                    curr_usage_df = usage_df[
                        (usage_df["latency_weight"] == lw)
                        & (usage_df["model_name"] == model)
                        & (usage_df["device_cpus"] == dev_cpus)
                        & (usage_df["edge_cpus"] == edge_cpus)
                    ]
                    curr_ax.plot(
                        curr_usage_df["max_noises"].unique(),
                        curr_usage_df["run_time"]
                        .groupby(curr_usage_df["max_noises"])
                        .mean(),
                        color="blue",
                        label="Real Latency",
                        marker="o",
                    )

                    curr_ax.set_title(f"Latency Weight: {lw}", fontsize=11)
                    curr_ax.set_xlabel("Max Noise")
                    curr_ax.set_ylabel("Time [s]")
                    curr_ax.legend()

                    curr_ax.set_xticks(curr_plan_df["max_noises"].unique())
                    curr_ax.set_xticklabels(curr_ax.get_xticklabels(), rotation=45)

                energy_weights = np.sort(plan_df["energy_weight"].unique())[::-1]

                for _, (ew, curr_ax) in enumerate(zip(energy_weights, axes[1])):
                    curr_plan_df = plan_df[
                        (plan_df["energy_weight"] == ew)
                        & (plan_df["model_name"] == model)
                        & (plan_df["device_cpus"] == dev_cpus)
                        & (plan_df["edge_cpus"] == edge_cpus)
                    ]
                    curr_ax.plot(
                        curr_plan_df["max_noises"].unique(),
                        curr_plan_df["energy_value"]
                        .groupby(curr_plan_df["max_noises"])
                        .mean(),
                        color="orange",
                        label="Planned Energy",
                        marker="o",
                    )

                    curr_ax.set_title(f"Energy Weight: {ew}", fontsize=11)
                    curr_ax.set_xlabel("Max Noise")
                    curr_ax.set_ylabel("Energy [J]")
                    curr_ax.legend()

                    curr_ax.set_xticks(curr_plan_df["max_noises"].unique())
                    curr_ax.set_xticklabels(curr_ax.get_xticklabels(), rotation=45)

                for ax in fig.get_axes():
                    ax.tick_params(labelleft=True)  # show tick labels
                    ax.yaxis.set_tick_params(which="both", labelleft=True)  # for sa

                fig.suptitle(
                    f"Device {dev_cpus} + Edge {edge_cpus} + Cloud ; Model: {model}",
                    fontsize=13,
                )

                plt.tight_layout()
                plt.savefig(
                    f"../Images/Pred_Comparisons/DeviceEdgeCloud/device_edge_cloud_plan_comparison_{model}_{dev_cpus}_{edge_cpus}.png"
                )


def main():
    plan_vs_real_comparison()

    pass


if __name__ == "__main__":
    main()
