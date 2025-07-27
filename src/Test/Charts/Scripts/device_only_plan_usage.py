import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import markers


def main():

    group_columns = [
        "model_name",
        "latency_weight",
        "energy_weight",
        "device_max_energy",
        "max_noises",
        "device_cpus",
    ]

    usage_df = pd.read_csv("../../Results/DeviceOnlyPlan/device_only_plan_Usage.csv")
    # usage_df = usage_df[
    #     (usage_df["model_name"] == "yolo11m")
    #     | (usage_df["model_name"] == "yolo11x-seg")
    # ]
    # usage_df = (
    #     usage_df.groupby(group_columns)["run_time"].agg(["mean", "std"]).reset_index()
    # )
    # usage_df.rename(columns={"mean": "mean_real", "std": "std_real"}, inplace=True)

    plan_df = pd.read_csv(
        "../../Results/DeviceOnlyPlan/device_only_plan_Generation.csv"
    )
    # plan_df = plan_df[
    #     (plan_df["model_name"] == "yolo11m") | (plan_df["model_name"] == "yolo11x-seg")
    # ]
    # plan_df = (
    #     plan_df.groupby(group_columns)["latency_value"]
    #     .agg(["mean", "std"])
    #     .reset_index()
    # )
    # plan_df.rename(columns={"mean": "mean_plan", "std": "std_plan"}, inplace=True)

    # merged_df = pd.merge(usage_df, plan_df, on=group_columns, how="inner")

    sns.set_style("whitegrid")  # Altri: "darkgrid", "white", "dark", "ticks"
    fig, axes = plt.subplots(figsize=(14, 7), nrows=2, ncols=1)

    models = ["yolo11m", "yolo11x-seg"]
    for i in range(len(axes)):
        model = models[i]
        ax = axes[i]

        sns.lineplot(
            data=usage_df[
                (usage_df["model_name"] == model) & (usage_df["device_cpus"] == 0.5)
            ],
            x="max_noises",
            y="run_time",
            ax=ax,
            label="Real Usage",
        )
        sns.lineplot(
            data=plan_df[
                (plan_df["model_name"] == model) & (plan_df["device_cpus"] == 0.5)
            ],
            x="max_noises",
            y="latency_value",
            ax=ax,
            label="Plan Usage",
        )

        ax.set_xticks(ticks=usage_df["max_noises"].unique())
        ax.set_xlabel("Max Noises")

        ax.set_ylabel("Time [s]")

        ax.set_title(f"Modello : {model}")

    fig.suptitle("Device Only Plan -- Max Noise vs Time")

    plt.tight_layout()
    plt.savefig("../Images/device_only_plan_Usage_max_noises.png")

    # pass


if __name__ == "__main__":
    main()
