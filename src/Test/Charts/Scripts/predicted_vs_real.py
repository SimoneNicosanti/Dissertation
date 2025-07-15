import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    device_exec_time_df = pd.read_csv(
        "../../Results/Exec_Profile_Time/Client_exec_profile_time.csv"
    )
    edge_exec_time_df = pd.read_csv(
        "../../Results/Exec_Profile_Time/Edge_exec_profile_time.csv"
    )

    profile_exec_time_df = pd.concat(
        [device_exec_time_df, edge_exec_time_df], ignore_index=True
    )

    grouped_predicted_df = (
        profile_exec_time_df.groupby(["model_name", "cpus"])["pred_time"]
        .agg(["mean", "std"])
        .reset_index()
    ).rename(columns={"mean": "mean_pred", "std": "std_pred"})

    real_df = pd.read_csv("../../Results/DeviceOnlyModel/device_only_model.csv")
    grouped_real_df = (
        real_df.groupby(["model_name", "cpus"])["run_time"]
        .agg(["mean", "std"])
        .reset_index()
    ).rename(columns={"mean": "mean_real", "std": "std_real"})

    merged_df = pd.merge(
        grouped_predicted_df, grouped_real_df, on=["model_name", "cpus"], how="inner"
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
        value_vars=["mean_real", "mean_pred"],
        var_name="Tipo",
        value_name="Valore",
    )

    # rinomino le categorie per chiarezza
    df_long["Tipo"] = df_long["Tipo"].map(
        {"mean_real": "Real", "mean_pred": "Predicted"}
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
            fontsize=10,
            color="black",
        )

    ax.set_title("Confronto Tempi Medi - Reali VS Predetti")
    ax.set_ylabel("Tempo [s]")
    ax.set_xlabel("Modello + CPUs")
    plt.legend(title="Caso")
    plt.tight_layout()

    plt.savefig("../Images/device_predicted_vs_real.png")
    pass


if __name__ == "__main__":
    main()
