import json
import os
from pathlib import Path
import subprocess

directory_dict = {
    "registry" : [
        "../proto_compiled/", 
        "../Registry/", "../StatePool/", 
        "../Main/RegistryMain.py", "../start.sh", "../Common/"
    ],
    "optimizer" : [
        "../proto_compiled/", 
        "../Optimizer/", 
        "../Main/OptimizerMain.py", "../start.sh", 
        "../Common/", "../CommonProfile/", "../CommonPlan/"
    ],
    "model-manager" : [
        "../proto_compiled/", 
        "../ModelPool/", "../ModelManager/",
        "../Main/ModelManagerMain.py", "../start.sh", 
        "../Common/", "../CommonProfile/", "../CommonPlan/", 
    ],
    "device" : [
        "../proto_compiled/", 
        "../Client/", "../Server/", "../FrontEnd/", 
        "../Main/ServerMain.py", "../Main/FrontEndMain.py", "../Main/ClientMain.py", "../start.sh", 
        "../Common/", "../CommonServer/",
        "../Other/latency_scripts/",
    ],
    "server-1" : [
        "../proto_compiled/", 
        "../Server/", 
        "../Main/ServerMain.py", "../start.sh", 
        "../Common/", "../CommonServer/",
    ],
    "server-2" :  [
        "../proto_compiled/", 
        "../Server/", 
        "../Main/ServerMain.py", "../start.sh", 
        "../Common/", "../CommonServer/",
    ],
}


def transfer_files(machine_name, machine_ip) :
    if machine_name not in directory_dict :
        return
    for dir in directory_dict[machine_name] :
        dest_dir = dir.replace("../", "")
        dest_dir = dest_dir.replace("Main/", "")
        print(dir, dest_dir)
        command = "rsync -e 'ssh -o StrictHostKeyChecking=accept-new' -avzu --progress --mkpath --recursive --exclude '*.pyc' {} google@{}:~/src/{}".format(dir, machine_ip, dest_dir)
        os.system(command)

def copy_config(machine_ip) :
    command = "rsync -e 'ssh -o StrictHostKeyChecking=accept-new' -avzu --progress --mkpath --recursive --exclude '*.pyc' ../config/deploy_config.ini google@{}:~/src/config/config.ini".format(machine_ip)
    os.system(command)

def copy_docker_config(machine_ip) :
    command = "rsync -e 'ssh -o StrictHostKeyChecking=accept-new' -avzu --progress --mkpath --recursive --exclude '*.pyc' ../docker/server.dockerfile google@{}:~/server.dockerfile".format(machine_ip)
    os.system(command)
    command = "rsync -e 'ssh -o StrictHostKeyChecking=accept-new' -avzu --progress --mkpath --recursive --exclude '*.pyc' ./container_start.py google@{}:~/container_start.py".format(machine_ip)
    os.system(command)

def get_nat_ips() :
    command = ['gcloud', 'compute', 'instances', 'list', '--format', 'json(name, networkInterfaces[].accessConfigs[].natIP)']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    instances = json.loads(result.stdout)

    name_ip_map = {}
    # Mostra l'output in formato dizionario
    for item in instances :
        name = item['name']
        if name in directory_dict :
            ip_addr = item['networkInterfaces'][0]['accessConfigs'][0]['natIP']
            name_ip_map[name] = ip_addr
    
    return name_ip_map

def main() :
    os.system("rm ~/.ssh/known_hosts")
    name_ip_map : dict[str, str] = get_nat_ips()
    print(name_ip_map)
    for machine_name, machine_ip in name_ip_map.items() :
        print("Sending to ", machine_name, machine_ip)
        copy_config(machine_ip)
        transfer_files(machine_name, machine_ip)
        pass

    for machine_name, machine_ip in name_ip_map.items() :
        if machine_name.startswith("server") or machine_name.startswith("device") :
            copy_docker_config(machine_ip)
    pass



if __name__ == "__main__" :
    main()