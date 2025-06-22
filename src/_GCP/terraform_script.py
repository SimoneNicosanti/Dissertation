#!/usr/bin/env python3

import argparse
import os

BASE_COMMAND = "terraform -chdir=./Terraform/ "

def main() :
    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--case", type=str, help="Apply or Destroy", required=True, choices=["a", "d"])
    parser.add_argument("--gpu", type=bool, help="Use GPU on Manager", default=False)


    args = parser.parse_args()

    command = BASE_COMMAND
    if args.case == "a":
        command += " apply"
        if args.gpu:
            command += """ -var="enable_gpu=true" """
        
    else:
        command += " destroy"
    
    # trunk-ignore(bandit/B605)
    os.system(command)


if __name__ == "__main__":
    main()