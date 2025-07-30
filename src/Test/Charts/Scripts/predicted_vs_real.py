import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    models = ["yolo11n-cls", "yolo11m", "yolo11x-seg", "yolo11x-seg_quant"]
    server_configs = [("Client", 0.5), ("Edge", 1.0), ("Cloud", 0.0)]

    pred_df = pd.DataFrame(
        columns=["model_name", "server", "cpus", "pred_avg_time", "interp_avg_time"]
    )

    layer_list = [
        "/model.1/conv/Conv",
        "/model.2/cv2/conv/Conv",
        "/model.3/conv/Conv",
        "/model.4/cv2/conv/Conv",
        "/model.5/conv/Conv",
        "/model.16/cv1/conv/Conv",
        "/model.17/conv/Conv",
        "/model.23/proto/cv1/conv/Conv",
        "/model.23/cv4.0/cv4.0.0/conv/Conv",
        "/model.23/cv2.0/cv2.0.0/conv/Conv",
        "/model.23/proto/upsample/ConvTranspose",
        "/model.23/proto/cv2/conv/Conv",
    ]

    profile_file = "../../Results/Exec_Profile/Client_yolo11x-seg_0.5_exec_profile.json"
    with open(profile_file, "r") as f:
        exec_profile_dict = json.load(f)

        interp_avg_time = 0
        for key in exec_profile_dict.keys():

            if key in layer_list:
                metric_name = "q_avg_time"
            else:
                metric_name = "nq_avg_time"

            tot_avg_time = exec_profile_dict["TotalSum"][metric_name]
            whole_model_avg_time = exec_profile_dict["WholeModel"][metric_name]

            if key != "WholeModel" and key != "TotalSum":
                interp_value = (
                    exec_profile_dict[key][metric_name] / tot_avg_time
                ) * whole_model_avg_time

                interp_avg_time += interp_value

    print("Interpolation Time >> ", interp_avg_time)
    return

    for model in models:
        if model.endswith("_quant"):
            metric_name = "q_avg_time"
            file_model_name = model.replace("_quant", "")
        else:
            metric_name = "nq_avg_time"
            file_model_name = model

        for config in server_configs:
            server, cpu = config
            profile_file = f"../../Results/Exec_Profile/{server}_{file_model_name}_{cpu}_exec_profile.json"
            if not os.path.exists(profile_file):
                continue

            with open(profile_file, "r") as f:
                exec_profile_dict = json.load(f)

                tot_avg_time = exec_profile_dict["TotalSum"][metric_name]
                whole_model_avg_time = exec_profile_dict["WholeModel"][metric_name]

                interp_avg_time = 0
                for key in exec_profile_dict.keys():
                    if key != "WholeModel" and key != "TotalSum":
                        interp_avg_time += (
                            exec_profile_dict[key][metric_name] / tot_avg_time
                        ) * whole_model_avg_time

            pred_df.loc[len(pred_df)] = [
                model,
                server,
                cpu,
                tot_avg_time,
                interp_avg_time,
            ]

    print(pred_df)

    real_df = pd.DataFrame(columns=["model_name", "server", "cpus", "real_avg_time"])
    for config in server_configs:
        server, cpus = config
        if server != "Client":
            continue
        profile_file = f"../../Results/DeviceOnlyModel/device_only_model.csv"
        if not os.path.exists(profile_file):
            continue
        profile_df = pd.read_csv(profile_file)

        avg_profile_df = (
            profile_df.groupby(["model_name", "cpus"])["run_time"].mean().reset_index()
        )

        for index, row in avg_profile_df.iterrows():
            model_name, cpus, avg_run_time = row
            real_df.loc[len(real_df)] = [model_name, server, cpus, avg_run_time]
    real_df = real_df[["model_name", "cpus", "real_avg_time"]]
    print(real_df)

    joined_df = pd.merge(pred_df, real_df, on=["model_name", "cpus"], how="inner")
    print(joined_df)

    df_long = joined_df.melt(
        id_vars=["model_name", "server", "cpus"],
        value_vars=["pred_avg_time", "real_avg_time", "interp_avg_time"],
        var_name="Case",
        value_name="Time [s]",
    )
    g = sns.catplot(
        data=df_long,
        x="model_name",
        y="Time [s]",
        hue="Case",
        col="cpus",
        kind="bar",
        height=4,
        aspect=1.5,
        palette="colorblind",
    )

    # Loop over each axis in the FacetGrid
    for ax in g.axes.flat:
        # Iterate over each bar container
        for container in ax.containers:
            # Add labels on top of each bar
            ax.bar_label(container, fmt="%.3f")  # format with 3 decimal places

    plt.savefig("../Images/predicted_vs_real.png")

    return


if __name__ == "__main__":
    main()
