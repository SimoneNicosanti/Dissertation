import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

GENERATION_PATH = "../../Results/DeviceEdgePlan/device_edge_plan_Generation.csv"
USAGE_PATH = "../../Results/DeviceEdgePlan/device_edge_plan_Usage.csv"


def plan_vs_real_comparison():
    plan_df = pd.read_csv(GENERATION_PATH)

    usage_df = pd.read_csv(USAGE_PATH)

    sns.set_style("whitegrid")  # Altri: "darkgrid", "white", "dark", "ticks"
    # fig, axes = plt.subplots(figsize=(14, 7), nrows=2, ncols=1)

    models = ["yolo11m"]
    for i in range(len(models)):
        model = models[i]

        for lw in plan_df["latency_weight"].unique():
            fig, axes = plt.subplots(figsize=(14, 7), nrows=2, ncols=1)

            sns.lineplot(
                data=plan_df[
                    (plan_df["model_name"] == model) & (plan_df["latency_weight"] == lw)
                ],
                x="max_noises",
                y="latency_value",
                ax=axes[0],
                palette="colorblind",
                hue="latency_weight",
            )

            sns.lineplot(
                data=usage_df[
                    (usage_df["model_name"] == model)
                    & (usage_df["latency_weight"] == lw)
                ],
                x="max_noises",
                y="run_time",
                ax=axes[1],
                palette="colorblind",
                hue="latency_weight",
            )

            axes[0].set_xticks(ticks=usage_df["max_noises"].unique())
            axes[1].set_xticks(ticks=usage_df["max_noises"].unique())
            # axes.set_xlabel("Max Noises")
            # axes.set_ylabel("Time [s]")

            plt.show()

        # ax.set_xticks(ticks=usage_df["max_noises"].unique())
        # ax.set_xlabel("Max Noises")

        # ax.set_ylabel("Time [s]")

        # ax.set_title(f"Modello : {model}")

    fig.suptitle("Device Only Plan -- Max Noise vs Time")

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

    models = ["yolo11m"]

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
    num_components_chart()
    pred_values_trends()
    plan_vs_real_comparison()

    pass


if __name__ == "__main__":
    main()
