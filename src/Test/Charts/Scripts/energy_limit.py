import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_consumption_line(dataframe: pd.DataFrame, max_value):
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(10, 4), nrows=1, ncols=1)
    ax.plot(
        dataframe["energy_limit"],
        dataframe["device_energy"],
        label="Device Energy",
        marker="o",
    )

    ax.hlines(
        max_value,
        dataframe["energy_limit"].min(),
        dataframe["energy_limit"].max(),
        colors="red",
        linestyles="--",
        label="Max Energy",
    )

    ax.set_xticks(ticks=dataframe["energy_limit"])
    ax.set_xticklabels(
        labels=[f"{x:.2f}" for x in dataframe["energy_limit"]], rotation=45
    )
    ax.set_xlabel("Energy Limit")
    ax.set_ylabel("Energy Consumption")
    ax.legend()
    fig.suptitle("Device Energy Consumption Trend")
    fig.tight_layout()

    plt.savefig("../Images/Energy_Limit/energy_limit_plot.png")
    pass


def plot_layers_bar_chart(dataframe: pd.DataFrame):
    df_melted = dataframe.melt(
        id_vars="energy_limit",
        value_vars=["device_layers", "edge_layers"],
        var_name="Server",
        value_name="Layers",
    )
    df_melted["Server"] = df_melted["Server"].replace(
        {"device_layers": "Device", "edge_layers": "Edge"}
    )

    df_melted["energy_limit"] = df_melted["energy_limit"].apply(lambda x: f"{x:.2f}")

    plt.figure(figsize=(15, 4))
    ax = sns.barplot(data=df_melted, x="energy_limit", y="Layers", hue="Server")

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2,  # centro della barra
                height + 0.1,  # leggermente sopra la sommità
                int(height),  # il numero da scrivere
                ha="center",
                va="bottom",
                fontsize=6,
            )

    plt.title("Layers distribution among Servers")
    plt.xticks(
        rotation=45,
    )
    plt.xlabel("Energy limit")
    plt.ylabel("Number of layers")
    plt.tight_layout()
    plt.savefig("../Images/Energy_Limit/energy_limit_layers.png")
    pass


def plot_components_bar_chart(dataframe: pd.DataFrame):
    df_melted = dataframe.melt(
        id_vars="energy_limit",
        value_vars=["device_comps", "edge_comps"],
        var_name="Server",
        value_name="Components",
    )
    df_melted["Server"] = df_melted["Server"].replace(
        {"device_comps": "Device", "edge_comps": "Edge"}
    )

    df_melted["energy_limit"] = df_melted["energy_limit"].apply(lambda x: f"{x:.2f}")

    plt.figure(figsize=(15, 4))
    ax = sns.barplot(data=df_melted, x="energy_limit", y="Components", hue="Server")

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2,  # centro della barra
                height + 0.1,  # leggermente sopra la sommità
                int(height),  # il numero da scrivere
                ha="center",
                va="bottom",
                fontsize=6,
            )

    plt.title("Layers distribution among Servers")
    plt.tight_layout()
    plt.xticks(
        rotation=45,
    )
    plt.xlabel("Energy limit")
    plt.ylabel("Number of layers")
    pass


def main():
    dataframe = pd.read_csv("../../Results/EnergyLimit/energy_limit.csv")

    max_energy_consumption = dataframe[dataframe["energy_limit"] == 0]["device_energy"]
    dataframe = dataframe[dataframe["energy_limit"] != 0]

    # print(dataframe)
    plot_consumption_line(dataframe, max_energy_consumption)
    plot_layers_bar_chart(dataframe)
    # plot_components_bar_chart(dataframe)
    pass


if __name__ == "__main__":
    main()
