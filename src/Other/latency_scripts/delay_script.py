import os
import argparse

def main() :

    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--dev", type=str, help="Interface Name", required=True)
    parser.add_argument("--classes", nargs="+", type=str, help="Delays for ips (bandwidth [Mbps], latency [ms], deviation [ms], handle)", required=True)
    parser.add_argument("--ips-info", nargs="+", type=str, help="Ip and Classes (ip addr, class handle)", required=True)
    args = parser.parse_args()

    dev = args.dev

    os.system("sudo iptables -F OUTPUT")
    # Parse degli argomenti

    os.system(f"tc qdisc del dev {dev} root || true")
    os.system(f"tc qdisc add dev {dev} handle 1: root htb")
    
    class_id = 15
    class_info : str
    for class_info in args.classes :
        param_list = class_info.split(",")
        cleaned_param_list = clean_input(param_list)

        print(cleaned_param_list)

        bandwidth, latency, deviation, handle = cleaned_param_list

        os.system(f"tc class add dev {dev} parent 1: classid 1:{class_id} htb rate {bandwidth}Mbit quantum 1500")
        os.system(f"tc qdisc add dev {dev} parent 1:{class_id} handle {handle} netem delay {latency}ms {deviation}ms distribution normal")
        os.system(f"tc filter add dev {dev} parent 1:0 prio 1 protocol ip handle {handle} fw flowid 1:{class_id}")
        
        class_id += 1
    
    ip_info : str
    for ip_info in args.ips_info :
        print(ip_info)
        param_list = ip_info.split(",")
        cleaned_param_list = clean_input(param_list)

        ip, handle = cleaned_param_list
        os.system(f"iptables -A OUTPUT -t mangle -d {ip} -j MARK --set-mark {handle}")

        


def clean_input(input_list : list[str]) -> list[str] :
    cleaned_input_list = []
    for elem in input_list :
        cleaned = elem.replace(" ", "")
        cleaned = cleaned.replace("\n", "")
        cleaned = cleaned.replace("(", "")
        cleaned = cleaned.replace(")", "")

        cleaned_input_list.append(cleaned)

    return cleaned_input_list


if __name__ == "__main__" :
    main()