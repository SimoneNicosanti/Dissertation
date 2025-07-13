import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    dataframe = pd.read_csv("../../Results/DeviceOnlyPlan/device_only_plan_Usage.csv")

    dataframe = dataframe[(dataframe["model_name"] == "yolo11x-seg")]
    # print(filtered_df)

    plt.figure(figsize=(15, 5))
    sns.set_style("whitegrid")  # Altri: "darkgrid", "white", "dark", "ticks"
    sns.set_palette("colorblind")  # Altre opzioni buone: "Set2", "colorblind", "husl"
    sns.lineplot(data=dataframe, x="max_noises", y="run_time", hue="device_cpus")

    plt.xticks(ticks=dataframe["max_noises"].unique())
    # plt.xlabel("Max Noises")
    # plt.ylabel("Run Time")
    # plt.title("Device Only Plan Usage")

    plt.show()
    pass


if __name__ == "__main__":
    main()
