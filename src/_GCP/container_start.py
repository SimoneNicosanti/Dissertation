#!/usr/bin/env python3

import argparse
import configparser
import os


def modify_energy_config(original_name: str, cpus: int = 0):

    if original_name != "edge" and original_name != "device":
        return

    config = configparser.ConfigParser()
    config.read("./src/config/energy_config.ini")

    base_energy_value = float(config.get("base_values", "COMP_ENERGY_PER_SEC"))

    if cpus == 0:
        cpus_multiplier = float(config.get("base_values", "DEFAULT_CORE_NUMS"))
    else:
        cpus_multiplier = cpus

    new_energy_value = base_energy_value * cpus_multiplier
    config.set(original_name, "COMP_ENERGY_PER_SEC", str(new_energy_value))

    with open("./src/config/energy_config.ini", "w") as f:
        config.write(f)


def map_execution_profiles(original_name: str, cpus: int = 0):

    directory = "/home/customuser/Exec_Profile/"

    orig_name_cap = original_name.capitalize()

    map_list = []

    if cpus == 0:
        find_part = "_0.0_"
    else:
        find_part = f"_{cpus}_"

    for file_name in os.listdir(directory):
        if (
            file_name.startswith(orig_name_cap + "_")
            and file_name.find(find_part) != -1
        ):
            dest_file_name = file_name.replace(orig_name_cap + "_", "")
            dest_file_name = dest_file_name.replace(
                f"{find_part}exec_profile.json", "_profile.json"
            )

            src_file_path = os.path.join(directory, file_name)
            dest_file_path = os.path.join(
                "/server_data/models_profiles", dest_file_name
            )

            map_list.append((src_file_path, dest_file_path))

    return map_list


def main():

    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--name", type=str, help="Name for Container", required=True)
    # parser.add_argument("--dockerfile", type=str, help="Dockerfile Name", required=True)
    parser.add_argument("--memory", type=float, help="Memory Size for Container")
    parser.add_argument("--cpus", type=float, help="CPUs number to set affinity to")
    parser.add_argument(
        "--gpu",
        help="Use GPU for Container",
        action="store_true",
    )
    parser.add_argument(
        "--connect", action="store_true", help="Connect to Existing Container"
    )

    # Parse degli argomenti
    args = parser.parse_args()

    original_name = args.name
    cont_name = args.name

    if cont_name == "edge" or cont_name == "device":
        cont_name = "server"

    if cont_name == "test":
        work_dir = "/src/Test/Scripts"
    else:
        work_dir = "/src"

    if args.connect:
        # trunk-ignore(bandit/B605)
        os.system(f"docker exec -it --workdir {work_dir} {cont_name} /bin/bash")
        return

    dockerfile_name = f"{cont_name}.dockerfile"

    # trunk-ignore(bandit/B605)
    os.system(f"docker stop {cont_name}")
    # trunk-ignore(bandit/B605)
    os.system(f"docker remove {cont_name}")
    # trunk-ignore(bandit/B605)
    os.system(f"docker build -t {cont_name}-image -f {dockerfile_name} .")

    ## Base Command
    command = f"docker run -it -d --name {cont_name} --network=host"

    ## Setting constraints
    if args.memory is not None:
        command += f" -m {args.memory}g"
    if args.cpus is not None:
        # cpus_list = [x for x in range(args.cpus)]
        # command += f" --cpuset-cpus={','.join(map(str, cpus_list))}"
        command += f" --cpus={args.cpus}"

    ## Setting Volumes
    command += " -v /home/customuser/src:/src"

    if cont_name == "manager" or cont_name == "test":
        command += " -v /home/customuser/models/:/model_pool_data/models/"
        command += (
            " -v /home/customuser/calibration/:/model_pool_data/calibration_dataset/"
        )
        command += (
            " -v /home/customuser/Model_Profile/:/model_profiler_data/models_profiles/"
        )
    if cont_name == "manager":
        command += " -v /home/customuser/layers/:/model_pool_data/layers/"

    ## Map Execution Profiles
    if original_name == "device" or original_name == "edge" or original_name == "cloud":
        if original_name == "device":
            use_name = "client"
        else:
            use_name = original_name

        for map_tuple in map_execution_profiles(
            use_name, args.cpus if args.cpus is not None else 0.0
        ):
            command += f" -v {map_tuple[0]}:{map_tuple[1]} "

    if args.gpu:
        command += " --gpus all "

    command += f" {cont_name}-image"
    print(command)
    # trunk-ignore(bandit/B605)
    os.system(command)

    if args.cpus is not None:
        modify_energy_config(original_name, args.cpus)
    else:
        modify_energy_config(original_name, 0)

    if args.gpu is True:
        os.system(f"docker exec -it {cont_name} pip install onnxruntime-gpu")

    # trunk-ignore(bandit/B605)
    os.system(f"docker exec -it --workdir {work_dir} {cont_name} /bin/bash")


if __name__ == "__main__":
    main()
