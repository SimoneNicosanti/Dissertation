#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
from pathlib import Path

BASE_COMMAND = "rsync -e 'ssh -o StrictHostKeyChecking=accept-new' -avzu --no-o --no-g --progress --mkpath --recursive --exclude '*.pyc' --exclude '*Out_*.jpg' --exclude '.npz' --exclude '*.tar' "


def copy_client(machine_ip):
    command = BASE_COMMAND + " ../Client/ customuser@{}:~/src/Client".format(machine_ip)
    # trunk-ignore(bandit/B605)
    os.system(command)
    return


def copy_common(machine_ip):
    dirs = [
        "Common",
        "CommonIds",
        "CommonModel",
        "CommonPlan",
        "CommonProfile",
        "CommonQuantization",
    ]
    for dir in dirs:
        command = BASE_COMMAND + " ../{}/ customuser@{}:~/src/{}".format(
            dir, machine_ip, dir
        )
        # trunk-ignore(bandit/B605)
        os.system(command)
    return


def copy_config(machine_ip):
    command = (
        BASE_COMMAND
        + " ../config/deploy_config.ini customuser@{}:~/src/config/config.ini".format(
            machine_ip
        )
    )
    # trunk-ignore(bandit/B605)
    os.system(command)

    command = (
        BASE_COMMAND
        + " ../config/energy_config.ini customuser@{}:~/src/config/energy_config.ini".format(
            machine_ip
        )
    )
    os.system(command)

    return


def copy_deployer(machine_ip):
    command = BASE_COMMAND + " ../Deployer/ customuser@{}:~/src/Deployer".format(
        machine_ip
    )
    # trunk-ignore(bandit/B605)
    os.system(command)
    return


def copy_docker(machine_ip):
    command = BASE_COMMAND + " ../docker/ customuser@{}:~/".format(machine_ip)
    # trunk-ignore(bandit/B605)
    os.system(command)

    command = (
        BASE_COMMAND
        + " ./container_start.py customuser@{}:~/container_start.py".format(machine_ip)
    )
    # trunk-ignore(bandit/B605)
    os.system(command)
    return


def copy_main(machine_ip):
    command = BASE_COMMAND + " ../Main/ customuser@{}:~/src".format(machine_ip)
    # trunk-ignore(bandit/B605)
    os.system(command)

    command = BASE_COMMAND + " ../start.sh customuser@{}:~/src/start.sh".format(
        machine_ip
    )
    # trunk-ignore(bandit/B605)
    os.system(command)

    command = (
        BASE_COMMAND
        + " ../_GCP/delay_script.py customuser@{}:~/delay_script.py".format(machine_ip)
    )
    # trunk-ignore(bandit/B605)
    os.system(command)
    return


def copy_manager(machine_ip):
    dirs = ["ModelDivider", "ModelPool", "ModelProfiler"]
    for dir in dirs:
        command = BASE_COMMAND + " ../{}/ customuser@{}:~/src/{}".format(
            dir, machine_ip, dir
        )
        # trunk-ignore(bandit/B605)
        os.system(command)
    return


def copy_optimizer(machine_ip):
    command = BASE_COMMAND + " ../Optimizer/ customuser@{}:~/src/Optimizer".format(
        machine_ip
    )
    # trunk-ignore(bandit/B605)
    os.system(command)
    return


def copy_other(machine_ip):
    command = (
        BASE_COMMAND
        + " ../Other/calibration_dataset/ customuser@{}:~/calibration".format(
            machine_ip
        )
    )
    # trunk-ignore(bandit/B605)
    os.system(command)
    return


def copy_proto(machine_ip):
    command = (
        BASE_COMMAND
        + " ../proto_compiled/ customuser@{}:~/src/proto_compiled".format(machine_ip)
    )
    # trunk-ignore(bandit/B605)
    os.system(command)
    return


def copy_registry(machine_ip):
    command = BASE_COMMAND + " ../Registry/ customuser@{}:~/src/Registry".format(
        machine_ip
    )
    # trunk-ignore(bandit/B605)
    os.system(command)
    return


def copy_server(machine_ip):
    command = BASE_COMMAND + " ../Server/ customuser@{}:~/src/Server".format(machine_ip)
    # trunk-ignore(bandit/B605)
    os.system(command)
    return


def copy_state_pool(machine_ip):
    command = BASE_COMMAND + " ../StatePool/ customuser@{}:~/src/StatePool".format(
        machine_ip
    )
    # trunk-ignore(bandit/B605)
    os.system(command)
    return


def copy_test(machine_ip):

    ## Remove results to avoid interference
    # trunk-ignore(bandit/B605)
    os.system(
        "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -q customuser@{} rm -r ~/src/Test/Results/*".format(
            machine_ip
        )
    )

    ## Sync again everything
    command = BASE_COMMAND + " ../Test/ customuser@{}:~/src/Test".format(machine_ip)

    # trunk-ignore(bandit/B605)
    os.system(command)
    return


def copy_model(machine_ip):
    command = BASE_COMMAND + " ../Other/models/ customuser@{}:~/models".format(
        machine_ip
    )
    # trunk-ignore(bandit/B605)
    os.system(command)
    return


def copy_profiles(machine_ip):
    command = (
        BASE_COMMAND
        + " ../Test/Results/Model_Profile/ customuser@{}:~/Model_Profile".format(
            machine_ip
        )
    )
    # trunk-ignore(bandit/B605)
    os.system(command)

    command = (
        BASE_COMMAND
        + " ../Test/Results/Exec_Profile/ customuser@{}:~/Exec_Profile".format(
            machine_ip
        )
    )
    # trunk-ignore(bandit/B605)
    os.system(command)

    return


directory_dict = {
    "Client": copy_client,
    "Common": copy_common,
    "Config": copy_config,
    "Deployer": copy_deployer,
    "Docker": copy_docker,
    "Main": copy_main,
    "Manager": copy_manager,
    "Optimizer": copy_optimizer,
    # "Other": copy_other,
    "Proto": copy_proto,
    "Registry": copy_registry,
    "Server": copy_server,
    "StatePool": copy_state_pool,
    "Test": copy_test,
    # "Model": copy_model,
    "Profiles": copy_profiles,
}


def get_nat_ips():
    command = [
        "gcloud",
        "compute",
        "instances",
        "list",
        "--format",
        "json(name, networkInterfaces[].accessConfigs[].natIP)",
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    instances = json.loads(result.stdout)

    name_ip_map = {}
    # Mostra l'output in formato dizionario
    for item in instances:
        print(item)
        name = item["name"]
        ip_addr = item["networkInterfaces"][0]["accessConfigs"][0]["natIP"]
        name_ip_map[name] = ip_addr

    return name_ip_map


def main():
    os.system("rm ~/.ssh/known_hosts")
    name_ip_map: dict[str, str] = get_nat_ips()
    print(name_ip_map)

    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--cases", nargs="+", type=str, help="Folder Cases", default=[])
    parser.add_argument(
        "--dests", nargs="+", type=str, help="Destination Names", default=[]
    )

    args = parser.parse_args()

    cases = args.cases
    dests = args.dests

    if len(cases) == 0:
        cases = directory_dict.keys()
    else:
        cases = [x for x in cases if x in directory_dict.keys()]

    if len(dests) == 0:
        dests = name_ip_map.keys()
    else:
        dests = [x for x in dests if x in name_ip_map.keys()]

    for dest_name in dests:
        machine_ip = name_ip_map[dest_name]

        for key in cases:
            print("↗️ Copying {} to {}".format(key, machine_ip))
            directory_dict[key](machine_ip)


if __name__ == "__main__":
    main()
