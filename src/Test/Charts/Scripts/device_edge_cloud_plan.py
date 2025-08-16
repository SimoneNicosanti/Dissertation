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
                (plan_df["latency_weight"] == lw) & (plan_df["model_name"] == model)
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
                (usage_df["latency_weight"] == lw) & (usage_df["model_name"] == model)
            ]
            curr_ax.plot(
                curr_usage_df["max_noises"].unique(),
                curr_usage_df["run_time"].groupby(curr_usage_df["max_noises"]).mean(),
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
                (plan_df["energy_weight"] == ew) & (plan_df["model_name"] == model)
            ]
            curr_ax.plot(
                curr_plan_df["max_noises"].unique(),
                curr_plan_df["energy_value"].groupby(curr_plan_df["max_noises"]).mean(),
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

        fig.suptitle(f"Device + Edge + Cloud ; Model: {model}", fontsize=13)

        plt.tight_layout()
        plt.savefig(
            f"../Images/Pred_Comparisons/device_edge_cloud_plan_comparison_{model}.png"
        )


def num_components_chart():
    plan_df = pd.read_csv(GENERATION_PATH)
    plan_df = (
        plan_df.groupby(
            [
                "model_name",
                "energy_weight",
                "latency_weight",
                "device_max_energy",
                "max_noises",
                "device_cpus",
                "edge_cpus",
                "device_bandwidth",
                "edge_bandwidth",
            ]
        )[["device_components", "edge_components"]]
        .mean()
        .reset_index()
    )
    plan_df["device_components"] = plan_df["device_components"] - 2
    df_long = plan_df.melt(
        id_vars=[
            "model_name",
            "energy_weight",
            "latency_weight",
            "device_max_energy",
            "max_noises",
            "device_cpus",
            "edge_cpus",
            "device_bandwidth",
            "edge_bandwidth",
        ],
        value_vars=["device_components", "edge_components"],
        var_name="device_type",
        value_name="comps_num",
    )

    sns.set_style("whitegrid")

    models = ["yolo11x-seg"]

    for _, model in enumerate(models):
        model_df = df_long[df_long["model_name"] == model]

        g = sns.catplot(
            data=model_df,
            x="max_noises",
            y="comps_num",
            hue="device_type",
            col="latency_weight",
            col_wrap=3,
            palette="colorblind",
            kind="bar",
            # facet_kws={"sharey": False, "sharex": False},
            height=3.5,  # ↓ decrease from default (usually 5)
            aspect=1.5,  # width/height ratio (optional)
        )

        plt.show()

    pass


def pred_values_trends():
    df = pd.read_csv(GENERATION_PATH)

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

    group_df = (
        df.groupby(group_cols)[["latency_value", "energy_value"]]
        .agg(["mean"])
        .reset_index()
    )
    group_df.columns = ["_".join(col).strip("_") for col in group_df.columns]

    models = ["yolo11m", "yolo11x-seg"]

    sns.set_style("whitegrid")

    for _, model in enumerate(models):
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        model_df = group_df[group_df["model_name"] == model]

        sns.lineplot(
            data=model_df,
            x="latency_weight",
            y="latency_value_mean",
            markers=True,
            palette="colorblind",
            ax=axes[0],
            hue="max_noises",
        )
        sns.lineplot(
            data=model_df,
            x="latency_weight",
            y="energy_value_mean",
            markers=True,
            palette="colorblind",
            ax=axes[1],
            hue="max_noises",
        )

        axes[0].set_ylabel("Latency [s]")
        axes[1].set_ylabel("Energy [J]")
        axes[0].set_xlabel("Latency Weight")
        axes[1].set_xlabel("Latency Weight")

        fig.suptitle("Latency and Energy Trends for " + model)
        plt.tight_layout()
        plt.savefig("../Images/" + model + "_planned_latency_energy_trends.png")


def main():
    # num_components_chart()
    # pred_values_trends()
    plan_vs_real_comparison()

    pass


if __name__ == "__main__":
    main()
