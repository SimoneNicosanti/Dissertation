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
    plan_df = plan_df[group_columns + ["latency_value"]]

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
            ci=None,
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
