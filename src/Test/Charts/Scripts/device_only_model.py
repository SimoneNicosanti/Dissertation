import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    dataframe: pd.DataFrame = pd.read_csv(
        "../../Results/DeviceOnlyModel/device_only_model.csv"
    )

    # dataframe = dataframe.groupby(["model_name", "cpus"], group_keys=False).head(50)

    stats_df = (
        dataframe.groupby(["model_name", "cpus"])["run_time"]
        .agg(["mean", "std"])
        .reset_index()
    )
    print(stats_df)

    model_sort = ["yolo11n-cls", "yolo11m", "yolo11x-seg"]
    dataframe["model_name"] = pd.Categorical(
        dataframe["model_name"], categories=model_sort, ordered=True
    )

    sns.set_theme(style="whitegrid")

    # Crea il grafico
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=dataframe,
        x="model_name",
        y="run_time",
        hue="cpus",  # gruppi interni alle barre
        palette="colorblind",
        errorbar=("ci", 95),  # disabilitiamo CI di seaborn per usare std personalizzata
        capsize=0.25,
    )

    for p in ax.patches:
        height = p.get_height()
        if height == 0:
            continue  # salta barre con altezza zero
        x = p.get_x() + p.get_width() / 2
        ax.text(
            x,
            height + 0.03,  # piccolo offset verticale sopra la barra
            f"{height:.3f}",  # formatta a 3 decimali
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )

    ax.set_title("Tempo Medio al Variare di --cpus (con CI 95%)")
    ax.set_ylabel("Tempo Medio [s]")
    ax.set_xlabel("Nome del Modello")
    plt.legend(title="--cpus")
    plt.tight_layout()

    plt.savefig("../Images/device_only_model.png")
    pass


if __name__ == "__main__":
    main()
