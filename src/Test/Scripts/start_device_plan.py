import argparse
import os
import subprocess
import time

## As the energy consumption is proportional to time and the execution is only on device, it does not matter which weights are used
## In this case we are mostly evaluating the impact of model quantization on execution time

## Only one test is needed, regardless of weights if noise == 0
## As there is linear relation between latency and energy, it is enough to run the test with a single couple when using only one device lw = 1.0, ew = 0.0

############################## YOLO11n-cls ##############################
# ./start_test.sh FullTest.py --model-name yolo11n-cls --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh FullTest.py --model-name yolo11n-cls --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.05 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1

# # ############################## YOLO11m-det ##############################
# ./start_test.sh FullTest.py --model-name yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh FullTest.py --model-name yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.025 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh FullTest.py --model-name yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.05 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh FullTest.py --model-name yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.075 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh FullTest.py --model-name yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.1 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh FullTest.py --model-name yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.125 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh FullTest.py --model-name yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.15 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh FullTest.py --model-name yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.175 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh FullTest.py --model-name yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.2 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh FullTest.py --model-name yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.5 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1


COMMAND = "./start_test.sh FullTest.py --model-name {} --latency-weight {} --energy-weight {} --device-max-energy {} --max-noises {} --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus {}"

MODELS = ["yolo11x-seg"]
LW = [1.0]
DEVICE_MAX_ENERGY = [0.0]
MAX_NOISES = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.5]


def main():
    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--device-cpus", type=float, help="Device CPUs", required=True)
    args = parser.parse_args()

    for model in MODELS:
        for lw in LW:
            ew = 1 - lw
            for device_max_energy in DEVICE_MAX_ENERGY:
                for max_noise in MAX_NOISES:
                    command = COMMAND.format(
                        model,
                        lw,
                        ew,
                        device_max_energy,
                        max_noise,
                        args.device_cpus,
                    )
                    print("")
                    print("🟢 Running >> ", command)
                    print("")

                    splitted_command = command.split(" ")
                    # trunk-ignore(bandit/B603)
                    subprocess.run(splitted_command)

                    time.sleep(1)

    pass


if __name__ == "__main__":
    main()
