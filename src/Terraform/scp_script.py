import json
import os
from pathlib import Path
import subprocess

directory_dict = {
    "registry" : ["../proto_compiled/", "../Registry/", "../StatePool/", "../Main/RegistryMain.py", "../start.sh", "../Common/"],
    "optimizer" : ["../proto_compiled/", "../Optimizer/", "../Main/OptimizerMain.py", "../start.sh", "../Common/"],
    "model-pool" : ["../proto_compiled/", "../ModelPool/", "../Main/ModelPoolMain.py", "../start.sh", "../Common/"],
    "device" : ["../proto_compiled/", "../Client/", "../Server/", "../FrontEnd/", "../CommonServer/", "../Main/ServerMain.py", "../Main/FrontEndMain.py", "../Main/ClientMain.py", "../start.sh", "../Common/"],
    "server-1" : ["../proto_compiled/", "../Server/", "../CommonServer/", "../Main/ServerMain.py", "../start.sh", "../Common/"],
    "server-2" : ["../proto_compiled/", "../Server/", "../CommonServer/", "../Main/ServerMain.py", "../start.sh", "../Common/"],
}


def transfer_files(machine_name, machine_ip) :
    if machine_name not in directory_dict :
        return
    for dir in directory_dict[machine_name] :
        dest_dir = dir.replace("../", "")
        dest_dir = dest_dir.replace("../Main/", "")
        print(dir, dest_dir)
        command = "rsync -e 'ssh -o StrictHostKeyChecking=accept-new' -avzu --progress --mkpath --recursive --exclude '*.pyc' {} google@{}:~/src/{}".format(dir, machine_ip, dest_dir)
        # print(command)
        # # command = "gcloud compute scp --recurse {} --zone europe-west12-c --project ai-at-edge-442215 google@{}:~/src".format(dir, machine_name)
        os.system(command)

# rsync -avz --progress --mkpath ../Other/models/ google@34.17.9.121:~/optimizer_data/models/

def copy_config(machine_ip) :
    command = "rsync -e 'ssh -o StrictHostKeyChecking=accept-new' -avzu --progress --mkpath --recursive --exclude '*.pyc' ../config/deploy_config.ini google@{}:~/src/config/config.ini".format(machine_ip)
    os.system(command)

# def copy_models() :
#     os.system("gcloud compute ssh --zone europe-west12-c --project ai-at-edge-442215 google@optimizer -- 'mkdir -p ~/optimizer_data/models && exit'")

#     model_dir = Path("../Other/models/")
#     for model_path in model_dir.rglob('*'): 
#         if model_path.name.find("yolo11n") == -1 :
#             continue
#         command = "gcloud compute scp {} --zone europe-west12-c --project ai-at-edge-442215 google@optimizer:~/optimizer_data/models/".format(model_path, model_path.name)
#         os.system(command)

def get_nat_ips() :
    command = ['gcloud', 'compute', 'instances', 'list', '--format', 'json(name, networkInterfaces[].accessConfigs[].natIP)']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    instances = json.loads(result.stdout)

    # Mostra l'output in formato dizionario
    name_ip_map = {item['name']: item['networkInterfaces'][0]['accessConfigs'][0]['natIP'] for item in instances}
    
    return name_ip_map

def main() :
    name_ip_map : dict[str, str] = get_nat_ips()
    print(name_ip_map)
    # copy_models()
    for machine_name, machine_ip in name_ip_map.items() :
        print("Sending to ", machine_name, machine_ip)
        copy_config(machine_ip)
        transfer_files(machine_name, machine_ip)
        pass
    pass



if __name__ == "__main__" :
    main()