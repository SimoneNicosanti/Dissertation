import pandas as pd
from matplotlib import pyplot as plt


def fix_layers_change_net_nodes(case: str):
    dataframe = pd.read_csv(f"../../Results/ScaleTest/{case}_scale_test.csv")

    dataframe = dataframe.groupby(
        ["num_nodes", "num_tensors", "net_nodes"], as_index=False
    ).mean()

    num_nodes = dataframe["num_nodes"].unique()

    max_charts = 5
    chart_idx = 0
    fig_num = 0
    for num_node in num_nodes:

        fig: plt.Figure
        axes: list[plt.Axes]
        if chart_idx == 0:
            fig, axes = plt.subplots(
                figsize=(14, 5), nrows=1, ncols=max_charts, sharey=True
            )
            fig.tight_layout(pad=2.5)  # default is 1.08
            # fig.tight_layout(rect=[0.1, 0.1, 0.1, 0.1])  # can also adjust the rectangle

        curr_df = dataframe[dataframe["num_nodes"] == num_node]

        axes[chart_idx].semilogy(
            curr_df["net_nodes"], curr_df["build_time"], label="Build Time", marker="o"
        )
        axes[chart_idx].semilogy(
            curr_df["net_nodes"],
            curr_df["latency_time"],
            label="Latency Time",
            marker="o",
        )
        axes[chart_idx].semilogy(
            curr_df["net_nodes"],
            curr_df["energy_time"],
            label="Energy Time",
            marker="o",
        )
        axes[chart_idx].semilogy(
            curr_df["net_nodes"], curr_df["whole_time"], label="Whole Time", marker="o"
        )
        axes[chart_idx].semilogy(
            curr_df["net_nodes"], curr_df["post_time"], label="Post Time", marker="o"
        )
        # axes[chart_idx].plot(
        #     curr_df["net_nodes"], curr_df["total_time"], label="Total Time", marker="o"
        # )

        axes[chart_idx].set_title("Num of Layers: " + str(num_node), fontsize=10)
        axes[chart_idx].set_xlabel("Num of Net Nodes")
        axes[chart_idx].set_xticks(curr_df["net_nodes"])
        axes[chart_idx].set_ylabel("Log Time [s]")
        axes[chart_idx].legend()

        chart_idx += 1
        if chart_idx == max_charts:
            plt.savefig(f"../Images/Scale_Test/{case}/Fix_Layers/fig_{fig_num}.png")
            fig_num += 1
            chart_idx = 0

    if chart_idx != 0:
        plt.savefig(f"../Images/Scale_Test/{case}/Fix_Layers/fig_{fig_num}.png")

    pass


def fix_net_nodes_change_layers(case: str):
    dataframe = pd.read_csv(f"../../Results/ScaleTest/{case}_scale_test.csv")

    dataframe = dataframe.groupby(
        ["num_nodes", "num_tensors", "net_nodes"], as_index=False
    ).mean()

    net_nodes = dataframe["net_nodes"].unique()

    max_charts = 3
    chart_idx = 0
    fig_num = 0
    for net_node in net_nodes:

        fig: plt.Figure
        axes: list[plt.Axes]
        if chart_idx == 0:
            fig, axes = plt.subplots(
                figsize=(14, 5), nrows=1, ncols=max_charts, sharey=True
            )
            fig.tight_layout(pad=2.5)  # default is 1.08
            # fig.tight_layout(rect=[0.1, 0.1, 0.1, 0.1])  # can also adjust the rectangle

        curr_df = dataframe[dataframe["net_nodes"] == net_node]

        axes[chart_idx].semilogy(
            curr_df["num_nodes"], curr_df["build_time"], label="Build Time", marker="o"
        )
        axes[chart_idx].semilogy(
            curr_df["num_nodes"],
            curr_df["latency_time"],
            label="Latency Time",
            marker="o",
        )
        axes[chart_idx].plot(
            curr_df["num_nodes"],
            curr_df["energy_time"],
            label="Energy Time",
            marker="o",
        )
        axes[chart_idx].semilogy(
            curr_df["num_nodes"], curr_df["whole_time"], label="Whole Time", marker="o"
        )
        axes[chart_idx].semilogy(
            curr_df["num_nodes"], curr_df["post_time"], label="Post Time", marker="o"
        )
        # axes[chart_idx].plot(
        #     curr_df["num_nodes"], curr_df["total_time"], label="Total Time", marker="o"
        # )

        axes[chart_idx].set_title("Num of Net Nodes: " + str(net_node), fontsize=10)
        axes[chart_idx].set_xlabel("Num of Layers")
        axes[chart_idx].set_xticks(curr_df["num_nodes"])
        axes[chart_idx].set_ylabel("Log Time [s]")
        axes[chart_idx].legend()

        chart_idx += 1
        if chart_idx == max_charts:
            plt.savefig(f"../Images/Scale_Test/{case}/Fix_Net_Nodes/fig_{fig_num}.png")
            fig_num += 1
            chart_idx = 0

    if chart_idx != 0:
        plt.savefig(f"../Images/Scale_Test/{case}/Fix_Net_Nodes/fig_{fig_num}.png")

    pass


def main():
    fix_layers_change_net_nodes(case="static")
    fix_net_nodes_change_layers(case="static")

    fix_layers_change_net_nodes(case="random")
    pass


if __name__ == "__main__":
    main()
