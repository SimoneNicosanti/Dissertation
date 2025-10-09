import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

GENERATION_PATH = "../../Results/DeviceEdgePlan/generation.csv"
USAGE_PATH = "../../Results/DeviceEdgePlan/usage.csv"


def plan_vs_real_comparison():

    group_cols = [
        "model_name",
        "latency_weight",
        "energy_weight",
        "device_max_energy",
        "max_noises",
        "device_cpus",
        "edge_cpus",
        "device_bandwidth",
        "edge_bandwidth",
    ]

    # ---- Planned times ----
    plan_df = pd.read_csv(GENERATION_PATH)
    plan_df = plan_df[group_cols + ["latency_value", "energy_value"]]

    # ---- Real times ----
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

                lw_list = plan_df["latency_weight"].unique()
                lw_list.sort()

                for idx, lw in enumerate(lw_list):  # lw_list :
                    curr_ax = axes[0][idx]

                    curr_usage_df = usage_df[
                        (usage_df["model_name"] == model)
                        & (usage_df["latency_weight"] == lw)
                        & (usage_df["device_cpus"] == dev_cpus)
                        & (usage_df["edge_cpus"] == edge_cpus)
                    ]

                    curr_ax.plot(
                        curr_usage_df["max_noises"].unique(),
                        curr_usage_df["run_time"]
                        .groupby(curr_usage_df["max_noises"])
                        .mean(),
                        marker="o",
                        color="blue",
                        label="Real Latency",
                    )

                    curr_plan_df = plan_df[
                        (plan_df["model_name"] == model)
                        & (plan_df["latency_weight"] == lw)
                        & (plan_df["device_cpus"] == dev_cpus)
                        & (plan_df["edge_cpus"] == edge_cpus)
                    ]

                    curr_ax.plot(
                        curr_plan_df["max_noises"].unique(),
                        curr_plan_df["latency_value"]
                        .groupby(curr_plan_df["max_noises"])
                        .mean(),
                        marker="o",
                        color="orange",
                        label="Planned Latency",
                    )

                    curr_ax.legend()

                    curr_ax.set_title(f"Latency Weight: {lw}", fontsize=11)

                    curr_ax.set_ylabel("Time [s]")
                    curr_ax.set_xlabel("Max Noise")

                    curr_ax.set_xticks(curr_usage_df["max_noises"].unique())
                    curr_ax.set_xticklabels(curr_ax.get_xticklabels(), rotation=45)

                    curr_plan_df = curr_plan_df.groupby(group_cols).mean().reset_index()
                    curr_usage_df = (
                        curr_usage_df.groupby(group_cols).mean().reset_index()
                    )

                    curr_comp_df = curr_usage_df.reset_index().merge(
                        curr_plan_df.reset_index(), on=group_cols, how="inner"
                    )[["max_noises", "run_time", "latency_value"]]

                    curr_comp_df.rename(
                        columns={
                            "max_noises": "\\textbf{Max Noise}",  # "Max Noise",
                            "run_time": "\\textbf{Run Time}",
                            "latency_value": "\\textbf{Plan Latency}",
                        },
                        inplace=True,
                    )

                    curr_comp_df["\\textbf{Latency Diff}"] = np.abs(
                        curr_comp_df["\\textbf{Run Time}"]
                        - curr_comp_df["\\textbf{Plan Latency}"]
                    )

                    curr_comp_df.to_latex(
                        f"../Csv/Pred_Comparisons/DeviceEdge/device_edge_plan_comparison_{model}_{dev_cpus}_{edge_cpus}_lw_{lw}.tex",
                        index=False,
                        column_format="|c|c|c|c|",
                        header=True,
                        float_format="%.4f",
                    )

                ew_list = np.sort(plan_df["energy_weight"].unique())[::-1]

                for idx, ew in enumerate(ew_list):  # ew_list :
                    curr_ax = axes[1][idx]

                    curr_plan_df = plan_df[
                        (plan_df["model_name"] == model)
                        & (plan_df["energy_weight"] == ew)
                        & (plan_df["device_cpus"] == dev_cpus)
                        & (plan_df["edge_cpus"] == edge_cpus)
                    ]

                    curr_ax.plot(
                        curr_plan_df["max_noises"].unique(),
                        curr_plan_df["energy_value"]
                        .groupby(curr_plan_df["max_noises"])
                        .mean(),
                        marker="o",
                        color="orange",
                        label="Planned Energy",
                    )

                    curr_ax.legend()

                    curr_ax.set_title(f"Energy Weight: {ew}", fontsize=11)

                    curr_ax.set_ylabel("Energy [J]")
                    curr_ax.set_xlabel("Max Noise")

                    curr_ax.set_xticks(curr_plan_df["max_noises"].unique())
                    curr_ax.set_xticklabels(curr_ax.get_xticklabels(), rotation=45)

                for ax in fig.get_axes():
                    ax.tick_params(labelleft=True)  # show tick labels
                    ax.yaxis.set_tick_params(which="both", labelleft=True)  # for sa

                fig.suptitle(
                    f"Device {dev_cpus} + Edge {edge_cpus} ; Model: {model}",
                    fontsize=13,
                )

                plt.tight_layout()
                plt.savefig(
                    f"../Images/Pred_Comparisons/DeviceEdge/device_edge_plan_comparison_{model}_{dev_cpus}_{edge_cpus}.svg"
                )


def main():
    plan_vs_real_comparison()

    pass


if __name__ == "__main__":
    main()
