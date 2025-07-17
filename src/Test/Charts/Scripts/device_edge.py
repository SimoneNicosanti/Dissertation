from cProfile import label

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

GENERATION_PATH = "../../Results/DeviceEdgePlan/device_edge_plan_Generation.csv"


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

    for i, model in enumerate(models):
        model_df = group_df[group_df["model_name"] == model]
        for max_noise in group_df["max_noises"].unique():
            _, axes = plt.subplots(1, 2, figsize=(15, 5))
            curr_df = model_df[model_df["max_noises"] == max_noise]

            sns.lineplot(
                data=curr_df,
                x="latency_weight",
                y="latency_value_mean",
                markers=True,
                palette="colorblind",
                ax=axes[0],
                label="Latency",
            )
            sns.lineplot(
                data=curr_df,
                x="latency_weight",
                y="energy_value_mean",
                markers=True,
                palette="colorblind",
                ax=axes[1],
                label="Energy",
            )

            plt.title("Max Noise: " + str(max_noise))
            plt.show()


def main():
    num_components_chart()
    pred_values_trends()

    pass


if __name__ == "__main__":
    main()
