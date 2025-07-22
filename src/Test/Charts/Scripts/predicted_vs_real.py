import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    models = ["yolo11n-cls", "yolo11m", "yolo11x-seg"]
    servers = ["Client", "Edge", "Cloud"]
    cpus = ["1", "2"]

    pred_df = pd.DataFrame(columns=["model_name", "server", "cpus", "pred_avg_time"])

    for server in servers:
        for model in models:
            for cpu in cpus:
                profile_file = f"../../Results/Exec_Profile/{server}_{model}_{cpu}_exec_profile.json"
                if not os.path.exists(profile_file):
                    continue
                with open(profile_file, "r") as f:
                    exec_profile_dict = json.load(f)

                    tot_avg_time = 0
                    for key in exec_profile_dict.keys():
                        tot_avg_time += exec_profile_dict[key]["nq_avg_time"]

                    # tot_avg_quant_time, tot_med_quant_time = model_exec_profile.get_total_quantized_time()
                    pred_df.loc[len(pred_df)] = [model, server, cpu, tot_avg_time]

    real_df = pd.DataFrame(columns=["model_name", "server", "cpus", "real_avg_time"])
    for server in servers:
        profile_file = f"../../Results/DeviceOnlyModel/{server}_only_model.csv"
        if not os.path.exists(profile_file):
            continue
        profile_df = pd.read_csv(profile_file)

        avg_profile_df = (
            profile_df.groupby(["model_name", "cpus"])["run_time"].mean().reset_index()
        )

        for index, row in avg_profile_df.iterrows():
            model_name, cpus, avg_run_time = row
            real_df.loc[len(real_df)] = [model_name, server, cpus, avg_run_time]

    pred_df["cpus"] = pred_df["cpus"].astype(int)
    real_df["cpus"] = real_df["cpus"].astype(int)

    joined_df = pd.merge(
        pred_df, real_df, on=["model_name", "server", "cpus"], how="inner"
    )
    print(joined_df)

    df_long = joined_df.melt(
        id_vars=["model_name", "server", "cpus"],
        value_vars=["pred_avg_time", "real_avg_time"],
        var_name="time_type",
        value_name="avg_time",
    )
    g = sns.catplot(
        data=df_long,
        x="model_name",
        y="avg_time",
        hue="time_type",
        col="cpus",
        row="server",
        kind="bar",
        height=4,
        aspect=1.5,
        palette="colorblind",
        ci=None,
    )

    # Loop over each axis in the FacetGrid
    for ax in g.axes.flat:
        # Iterate over each bar container
        for container in ax.containers:
            # Add labels on top of each bar
            ax.bar_label(container, fmt="%.3f")  # format with 3 decimal places

    plt.show()

    return

    real_df = pd.read_csv("../../Results/DeviceOnlyModel/device_only_model.csv")
    grouped_real_df = (
        real_df.groupby(["model_name", "cpus"])["run_time"]
        .agg(["mean", "median"])
        .reset_index()
    ).rename(columns={"mean": "mean_real", "median": "median_real"})

    merged_df = pd.merge(
        pred_df, grouped_real_df, on=["model_name", "cpus"], how="inner"
    )
    print(merged_df)

    merged_df["model_name"] = (
        merged_df["model_name"].str.replace("yolo11", "").replace("m", "m-det")
    )

    # creo una colonna per label gruppo
    merged_df["label"] = merged_df["model_name"] + "\n" + merged_df["cpus"].astype(str)

    # trasformo in formato lungo
    df_long = pd.melt(
        merged_df,
        id_vars=["label"],
        value_vars=["mean_real", "tot_avg_time", "median_real", "tot_med_time"],
        var_name="Tipo",
        value_name="Valore",
    )

    # rinomino le categorie per chiarezza
    df_long["Tipo"] = df_long["Tipo"].map(
        {
            "mean_real": "Avg Real",
            "median_real": "Med Real",
            "tot_avg_time": "Avg Predicted",
            "tot_med_time": "Med Predicted",
        }
    )

    sns.set_theme(style="whitegrid")

    # plot
    plt.figure(figsize=(18, 6))
    ax = sns.barplot(
        data=df_long,
        x="label",
        y="Valore",
        hue="Tipo",
        palette="colorblind",
    )

    for p in ax.patches:
        height = p.get_height()
        if height == 0:
            continue  # salta barre con altezza zero
        x = p.get_x() + p.get_width() / 2
        ax.text(
            x,
            height + 0.03,  # piccolo offset verticale sopra la barra
            f"{height:.2f}",  # formatta a 3 decimali
            ha="center",
            va="bottom",
            fontsize=6,
            color="black",
        )

    ax.set_title("Confronto Tempi Medi - Reali VS Predetti Media VS Predetti Mediana")
    ax.set_ylabel("Tempo [s]")
    ax.set_xlabel("Modello + CPUs")
    plt.legend(title="Caso")
    plt.tight_layout()

    plt.savefig("../Images/device_predicted_vs_real.png")
    pass


if __name__ == "__main__":
    main()
