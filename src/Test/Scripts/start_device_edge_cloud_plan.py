import argparse
import os
import subprocess
import time

COMMAND = "./start_test.sh DeviceEdgeCloudPlan.py --model {} --latency-weight {} --energy-weight {} --device-max-energy {} --max-noises {} --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus {} --edge-cpus {} --device-bandwidth {} --edge-bandwidth {} --cloud-bandwidth {}"


MODELS = ["yolo11x-seg"]
LW = [1.0, 0.75, 0.5, 0.25, 0.0]
DEVICE_MAX_ENERGYS = [0.0]
MAX_NOISES = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.5]


def main():
    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--device-cpus", type=float, help="Device CPUs", required=True)
    parser.add_argument("--edge-cpus", type=float, help="Edge CPUs", required=True)

    parser.add_argument(
        "--device-bandwidth", type=float, help="Device Bandwidth", required=True
    )
    parser.add_argument(
        "--edge-bandwidth", type=float, help="Edge Bandwidth", required=True
    )
    parser.add_argument(
        "--cloud-bandwidth", type=float, help="Cloud Bandwidth", required=True
    )

    args = parser.parse_args()

    for model in MODELS:
        for lw in LW:
            ew = 1 - lw
            for device_max_energy in DEVICE_MAX_ENERGYS:
                for max_noise in MAX_NOISES:
                    command = COMMAND.format(
                        model,
                        lw,
                        ew,
                        device_max_energy,
                        max_noise,
                        args.device_cpus,
                        args.edge_cpus,
                        args.device_bandwidth,
                        args.edge_bandwidth,
                        args.cloud_bandwidth,
                    )
                    print("")
                    print("🟢 Running >> ", command)
                    print("")

                    splitted_command = command.split(" ")
                    # trunk-ignore(bandit/B605)
                    # trunk-ignore(bandit/B603)
                    subprocess.run(splitted_command)

                    time.sleep(1)

    pass


if __name__ == "__main__":
    main()
