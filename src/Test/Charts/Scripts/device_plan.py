from turtle import color

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():

    group_columns = [
        "model_name",
        "latency_weight",
        "energy_weight",
        "device_max_energy",
        "max_noises",
        "device_cpus",
    ]

    usage_df = pd.read_csv("../../Results/DevicePlan/usage.csv")
    usage_df = usage_df[group_columns + ["run_time"]]

    plan_df = pd.read_csv("../../Results/DevicePlan/generation.csv")
    plan_df = plan_df[group_columns + ["latency_value", "energy_value"]]

    sns.set_style("whitegrid")  # Altri: "darkgrid", "white", "dark", "ticks"
    fig, axes = plt.subplots(figsize=(14, 7), nrows=2, ncols=1)

    models = ["yolo11x-seg"]
    for model in models:
        for cpus in usage_df["device_cpus"].unique():
            fig: plt.Figure
            axes: list[plt.Axes]
            fig, axes = plt.subplots(figsize=(14, 7), nrows=2, ncols=1)

            curr_usage_df = (
                usage_df[
                    (usage_df["model_name"] == model)
                    & (usage_df["device_cpus"] == cpus)
                ]
                .groupby(group_columns)
                .mean()
                .reset_index()
            )

            curr_plan_df = (
                plan_df[
                    (plan_df["model_name"] == model) & (plan_df["device_cpus"] == cpus)
                ]
                .groupby(group_columns)
                .mean()
                .reset_index()
            )

            axes[0].plot(
                usage_df["max_noises"].unique(),
                curr_usage_df["run_time"],
                label="Real Latency",
                color="blue",
                marker="o",
            )
            axes[0].plot(
                plan_df["max_noises"].unique(),
                curr_plan_df["latency_value"],
                label="Plan Latency",
                color="orange",
                marker="o",
            )
            axes[1].plot(
                plan_df["max_noises"].unique(),
                curr_plan_df["energy_value"],
                label="Plan Energy",
                color="orange",
                marker="o",
            )

            axes[0].set_xticks(ticks=usage_df["max_noises"].unique())
            axes[1].set_xticks(ticks=usage_df["max_noises"].unique())

            axes[0].set_xlabel("Max Noise")
            axes[1].set_xlabel("Max Noise")

            axes[0].set_ylabel("Time [s]")
            axes[1].set_ylabel("Energy [J]")

            axes[0].set_title("Latency")
            axes[1].set_title("Energy")

            axes[0].legend()
            axes[1].legend()

            fig.suptitle(f"Device CPUs {cpus} - Model {model}")

            plt.tight_layout()
            plt.savefig(
                f"../Images/Pred_Comparisons/Device/device_plan_comparison_{model}_{cpus}.png"
            )

    # pass


if __name__ == "__main__":
    main()
