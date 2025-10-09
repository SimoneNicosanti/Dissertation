import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_profile_sum(profile_dict: dict) -> dict:
    tot_sum = 0
    for layer_id in profile_dict.keys():
        if layer_id == "WholeModel" or layer_id == "MixedModel":
            continue
        tot_sum += profile_dict[layer_id]["nq_avg_time"]

    return tot_sum


def main():
    device_profile = json.load(
        open(
            "../../Results/Exec_Profile/Client_yolo11x-seg_1.0_exec_profile.json",
            "r",
        )
    )
    edge_profile = json.load(
        open("../../Results/Exec_Profile/Edge_yolo11x-seg_1.0_exec_profile.json", "r")
    )
    cloud_profile = json.load(
        open("../../Results/Exec_Profile/Cloud_yolo11x-seg_0.0_exec_profile.json", "r")
    )

    device_sum = compute_profile_sum(device_profile)
    edge_sum = compute_profile_sum(edge_profile)
    cloud_sum = compute_profile_sum(cloud_profile)

    device_real = device_profile["WholeModel"]["nq_avg_time"]
    edge_real = edge_profile["WholeModel"]["nq_avg_time"]
    cloud_real = cloud_profile["WholeModel"]["nq_avg_time"]

    sums = [device_sum, edge_sum, cloud_sum]
    reals = [device_real, edge_real, cloud_real]

    # Organize data
    rows = [
        {
            "Machine": "Device",
            "Sum": device_sum,
            "Real": device_real,
            "Difference": device_sum - device_real,
        },
        {
            "Machine": "Edge",
            "Sum": edge_sum,
            "Real": edge_real,
            "Difference": edge_sum - edge_real,
        },
        {
            "Machine": "Cloud",
            "Sum": cloud_sum,
            "Real": cloud_real,
            "Difference": cloud_sum - cloud_real,
        },
    ]

    # Write to CSV
    with open("../Csv/Real_vs_Sum/real_vs_sum.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Machine", "Sum", "Real", "Difference"])
        writer.writeheader()
        writer.writerows(rows)

    full_df = pd.read_csv("../Csv/Real_vs_Sum/real_vs_sum.csv")
    full_df.to_latex(
        "../Csv/Real_vs_Sum/real_vs_sum.tex",
        index=False,
        column_format="|c|c|c|c|",
        header=True,
        float_format="%.4f",
    )

    labels = ["Device", "Edge", "Cloud"]

    x = np.arange(len(labels))  # posizioni sull'asse x
    width = 0.35  # larghezza barre

    fig, ax = plt.subplots()

    bars1 = ax.bar(x - width / 2, sums, width, label="Layer Sum")
    bars2 = ax.bar(x + width / 2, reals, width, label="Real Model")

    # Etichette e titolo
    ax.set_ylabel("Time [s]")
    ax.set_xlabel("Machine")
    ax.set_title("Sum Layer Time vs Real Model Time - yolo11x-seg")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig("../Images/Real_vs_Sum/real_vs_sum.svg")


if __name__ == "__main__":
    main()
