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
        fig, axes = plt.subplots(figsize=(14, 7), nrows=2, ncols=1)

        sns.lineplot(
            data=usage_df[
                (usage_df["model_name"] == model) & (usage_df["device_cpus"] == 1.0)
            ],
            x="max_noises",
            y="run_time",
            ax=axes[0],
            label="Real Latency",
        )
        sns.lineplot(
            data=plan_df[
                (plan_df["model_name"] == model) & (plan_df["device_cpus"] == 1.0)
            ],
            x="max_noises",
            y="latency_value",
            label="Plan Latency",
            ax=axes[0],
            ci=None,
        )

        sns.lineplot(
            data=plan_df[
                (plan_df["model_name"] == model) & (plan_df["device_cpus"] == 1.0)
            ],
            x="max_noises",
            y="energy_value",
            label="Plan Energy",
            ax=axes[1],
            ci=None,
        )

        axes[0].set_xticks(ticks=usage_df["max_noises"].unique())
        axes[1].set_xticks(ticks=usage_df["max_noises"].unique())

        axes[0].set_xlabel("Max Noise")
        axes[1].set_xlabel("Max Noise")

        axes[0].set_ylabel("Time [s]")
        axes[1].set_ylabel("Energy [J]")

        axes[0].set_title("Latency")
        axes[1].set_title("Energy")

        fig.suptitle(f"Device - Model {model}")

        plt.tight_layout()
        plt.savefig(f"../Images/Pred_Comparisons/device_plan_comparison_{model}.png")

    # pass


if __name__ == "__main__":
    main()
