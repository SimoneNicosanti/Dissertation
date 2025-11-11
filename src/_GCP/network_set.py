#!/bin/bash

## https://news.ycombinator.com/item?id=44658973
## Must disable the cold start after idle: otherwise the predictions are completely broken
## Otherwise add a keep warm channel function

# gcloud compute ssh device --command="sudo python3 delay_script.py --dev ens3 --bandwidth 5.0 --latencies 5 55 --ips 10.0.1.16 10.0.1.17"
# gcloud compute ssh device --command="sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0"

# gcloud compute ssh edge --command="sudo python3 delay_script.py --dev ens3 --bandwidth 20.0 --latencies 5 50 --ips 10.0.1.15 10.0.1.17"
# gcloud compute ssh edge --command="sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0"

# gcloud compute ssh cloud --command="sudo python3 delay_script.py --dev ens5 --bandwidth 100.0 --latencies 55 50 --ips 10.0.1.15 10.0.1.16"
# gcloud compute ssh cloud --command="sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0"


import json
import os
import subprocess


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
    ip_dict = get_nat_ips()

    if "device" in ip_dict.keys():
        os.system(
            """ gcloud compute ssh device --command="sudo python3 delay_script.py --dev ens4 --bandwidth 5.0 --latencies 5 55 --ips 10.0.1.16 10.0.1.17" """
        )
        os.system(
            """ gcloud compute ssh device --command="sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0" """
        )

    if "edge" in ip_dict.keys():
        os.system(
            """ gcloud compute ssh edge --command="sudo python3 delay_script.py --dev ens3 --bandwidth 20.0 --latencies 5 50 --ips 10.0.1.15 10.0.1.17" """
        )
        os.system(
            """gcloud compute ssh edge --command="sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0" """
        )

    if "cloud" in ip_dict.keys():
        os.system(
            """ gcloud compute ssh cloud --command="sudo python3 delay_script.py --dev ens5 --bandwidth 100.0 --latencies 55 50 --ips 10.0.1.15 10.0.1.16" """
        )
        os.system(
            """ gcloud compute ssh cloud --command="sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0" """
        )
    pass


if __name__ == "__main__":
    main()
