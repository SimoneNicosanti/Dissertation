import os
from pathlib import Path

directory_dict = {
    # "registry" : ["../proto_compiled", "../Registry", "../StatePool", "../Main/RegistryMain.py", "../start.sh", "../Common"],
    # "optimizer" : ["../proto_compiled", "../Optimizer", "../Main/OptimizerMain.py", "../start.sh", "../Common"],
    # "model-pool" : ["../proto_compiled", "../ModelPool", "../Main/ModelPoolMain.py", "../start.sh", "../Common"],
    "device" : ["../proto_compiled", "../Client", "../Server", "../FrontEnd", "../CommonServer", "../Main/ServerMain.py", "../Main/FrontEndMain.py", "../Main/ClientMain.py", "../start.sh", "../Common"],
    "server-1" : ["../proto_compiled", "../Server", "../CommonServer", "../Main/ServerMain.py", "../start.sh", "../Common"],
    "server-2" : ["../proto_compiled", "../Server", "../CommonServer", "../Main/ServerMain.py", "../start.sh", "../Common"],
}


def transfer_files(machine_name) :
    for dir in directory_dict[machine_name] :
        command = "gcloud compute scp --recurse {} --zone europe-west12-c --project ai-at-edge-442215 google@{}:~/src".format(dir, machine_name)
        os.system(command)

def copy_config(machine_name) :
    config_path = "../config/deploy_config.ini"
    os.system("gcloud compute ssh --zone europe-west12-c --project ai-at-edge-442215 google@{} -- 'mkdir -p ~/src/config && exit'".format(machine_name))
    command = "gcloud compute scp {} --zone europe-west12-c --project ai-at-edge-442215 google@{}:~/src/config/config.ini".format(config_path, machine_name)
    os.system(command)

def copy_models() :
    os.system("gcloud compute ssh --zone europe-west12-c --project ai-at-edge-442215 google@optimizer -- 'mkdir -p ~/optimizer_data/models && exit'")

    model_dir = Path("../Other/models/")
    for model_path in model_dir.rglob('*'): 
        if model_path.name.find("yolo11n") == -1 :
            continue
        command = "gcloud compute scp {} --zone europe-west12-c --project ai-at-edge-442215 google@optimizer:~/optimizer_data/models/".format(model_path, model_path.name)
        os.system(command)

def main() :
    # copy_models()
    for machin_name in directory_dict.keys() :
        copy_config(machin_name)
        transfer_files(machin_name)
    pass



if __name__ == "__main__" :
    main()