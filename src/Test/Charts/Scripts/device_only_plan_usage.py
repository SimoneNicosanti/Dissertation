import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import markers


def main():
    dataframe = pd.read_csv("../../Results/DeviceOnlyPlan/device_only_plan_Usage.csv")

    # print(filtered_df)

    sns.set_style("whitegrid")  # Altri: "darkgrid", "white", "dark", "ticks"
    fig, axes = plt.subplots(figsize=(14, 7), nrows=2, ncols=1)

    sns.lineplot(
        data=dataframe[dataframe["model_name"] == "yolo11m"],
        x="max_noises",
        y="run_time",
        hue="device_cpus",
        ax=axes[0],
        palette="colorblind",
    )
    sns.lineplot(
        data=dataframe[dataframe["model_name"] == "yolo11x-seg"],
        x="max_noises",
        y="run_time",
        hue="device_cpus",
        ax=axes[1],
        palette="colorblind",
    )

    axes[0].set_xticks(ticks=dataframe["max_noises"].unique())
    axes[1].set_xticks(ticks=dataframe["max_noises"].unique())

    axes[0].set_xlabel("Max Noises")
    axes[1].set_xlabel("Max Noises")

    axes[0].set_ylabel("Run Time")
    axes[1].set_ylabel("Run Time")

    axes[0].set_title("Modello : yolo11m-det")
    axes[1].set_title("Modello : yolo11x-seg")

    fig.suptitle("Device Only Plan -- Max Noise vs Run Time")

    plt.tight_layout()
    plt.savefig("../Images/device_only_plan_Usage_max_noises.png")

    pass


if __name__ == "__main__":
    main()
