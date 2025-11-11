#!/usr/bin/env python3

import argparse
import os

BASE_COMMAND = "terraform -chdir=./Terraform/ "


def main():
    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument(
        "--case", type=str, help="Apply or Destroy", required=True, choices=["a", "d"]
    )
    parser.add_argument(
        "--only-optimizer", action="store_true", help="Use only optimizer"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU on Manager")
    parser.add_argument("--device", action="store_true", help="Activate Device")
    parser.add_argument("--edge", action="store_true", help="Activate Edge")
    parser.add_argument("--cloud", action="store_true", help="Activate Cloud")

    args = parser.parse_args()

    command = BASE_COMMAND
    if args.case == "a":
        command += " apply"

        if args.gpu:
            command += """ -var="enable_gpu=true" """
        if args.device:
            command += """ -var="enable_device=true" """
        if args.edge:
            command += """ -var="enable_edge=true" """
        if args.cloud:
            command += """ -var="enable_cloud=true" """
        if args.only_optimizer:
            command += """ -var="only_optimizer=true" """

    else:
        command += " destroy"

    # trunk-ignore(bandit/B605)
    os.system(command)


if __name__ == "__main__":
    main()
