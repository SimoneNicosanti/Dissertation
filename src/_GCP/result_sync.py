#!/usr/bin/env python3

import argparse
import json
import os
import subprocess

BASE_COMMAND = "rsync -e 'ssh -o StrictHostKeyChecking=accept-new' -avzu --no-o --no-g --progress --mkpath --recursive "


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Source Machine Name", required=True)

    args = parser.parse_args()
    src_name = args.src

    if src_name not in name_ip_map.keys():
        print("Machine not found")
        return

    machine_ip = name_ip_map[src_name]

    command = BASE_COMMAND + f" customuser@{machine_ip}:~/src/Test/Results/ ../Test/Results"

    os.system(command)

    pass


if __name__ == "__main__":
    main()
