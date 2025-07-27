import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

GENERATION_PATH = "../../Results/DeviceEdgePlan/device_edge_plan_Generation.csv"
USAGE_PATH = "../../Results/DeviceEdgePlan/device_edge_plan_Usage.csv"


def plan_vs_real_comparison():
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
        )["latency_value"]
        .mean()
        .reset_index()
    ).rename(columns={"latency_value": "plan_latency"})

    usage_df = pd.read_csv(USAGE_PATH)
    usage_df = (
        usage_df.groupby(
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
        )["run_time"]
        .mean()
        .reset_index()
    ).rename(columns={"run_time": "real_time"})

    merged_df = plan_df.merge(
        usage_df,
        on=[
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
        how="inner",
    )

    df_long = pd.melt(
        merged_df,
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
        value_vars=["plan_latency", "real_time"],
        var_name="time_type",
        value_name="time_value",
    )

    sns.set_style("whitegrid")  # Altri: "darkgrid", "white", "dark", "ticks"

    models = ["yolo11x-seg"]
    for i in range(len(models)):
        model = models[i]
        # fig, axes = plt.subplots(figsize=(14, 7), nrows=5, ncols=2)

        g = sns.relplot(
            data=df_long[df_long["model_name"] == model],
            x="max_noises",
            y="time_value",
            hue="time_type",
            col="latency_weight",
            col_wrap=3,
            palette="colorblind",
            kind="line",
            facet_kws={"sharey": False, "sharex": False},
            height=3.5,  # ↓ decrease from default (usually 5)
            aspect=1.5,  # width/height ratio (optional)
        )

        xticks = df_long[df_long["model_name"] == model]["max_noises"].unique()
        for ax in g.axes.flat:
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{x}" for x in xticks], rotation=45)

    # fig.suptitle("Device Only Plan -- Max Noise vs Time")

    plt.tight_layout()
    plt.show()


def num_components_chart():
    df = pd.read_csv(GENERATION_PATH)

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

    plt.clf()


def main():
    # num_components_chart()
    pred_values_trends()
    plan_vs_real_comparison()

    pass


if __name__ == "__main__":
    main()
