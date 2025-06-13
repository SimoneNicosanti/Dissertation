#!/usr/bin/env python3

import argparse
import os


def main():

    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--name", type=str, help="Name for Container", required=True)
    # parser.add_argument("--dockerfile", type=str, help="Dockerfile Name", required=True)
    parser.add_argument("--memory", type=float, help="Memory Size for Container")
    parser.add_argument("--cpus", type=float, help="CPUs percentage for Container")

    # Parse degli argomenti
    args = parser.parse_args()
    cont_name = args.name

    dockerfile_name = f"{cont_name}.dockerfile"

    os.system(f"docker stop {cont_name}")
    os.system(f"docker remove {cont_name}")
    os.system(f"docker build -t {cont_name}-image -f {dockerfile_name} .")

    ## Base Command
    command = f"docker run -it -d --name {cont_name} --network host"

    ## Setting constraints
    if args.memory is not None:
        command += f" -m {args.memory}g"
    if args.cpus is not None:
        command += f" --cpus={args.cpus}"

    ## Setting Volumes
    command += " -v /home/customuser/src:/src"

    if cont_name == "manager":
        command += " -v /home/customuser/models/:/model_pool_data/models/"
        command += (
            " -v /home/customuser/calibration/:/model_pool_data/calibration_dataset/"
        )
        command += " --gpus all "

    command += f" {cont_name}-image"
    print(command)
    os.system(command)

    os.system(f"docker exec -it --workdir /src {cont_name} /bin/bash")


if __name__ == "__main__":
    main()
