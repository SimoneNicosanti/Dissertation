import os
import argparse




def main() :

    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--dev", type=str, help="Interface Name", required=True)
    parser.add_argument("--bandwidth", type=float, help="Bandwidth [MBps]", default=10_000)
    parser.add_argument("--latencies", nargs = "+", type=float, help="Latencies To Add")
    parser.add_argument("--ips", nargs = "+", type=str, help="Destination ips")
    args = parser.parse_args()

    

    dev = args.dev
    bandwidth = args.bandwidth
    destinations = args.ips
    latencies = args.latencies
    
    # Clearing up root qdisc
    os.system(f"tc qdisc del dev {dev} root || true")

   
    ## Creating default class for unclassified packets
    ## All unclassified packets will go here with limited bandwidth
    ## Default bandwidth is 10 Gbps, otherwise it will be limited as specified
    os.system(f"tc qdisc add dev {dev} root handle 1:0 htb default 1")

    ## Default class with bandwidth limit
    os.system(f"tc class add dev {dev} parent 1:0 classid 1:1 htb rate {bandwidth}mbps ceil {bandwidth}mbps")

    if latencies is not None and destinations is not None :

        if len(latencies) != len(destinations) :
            raise ValueError("Latencies and Destinations must have the same length")

        class_idx = 10
        for i in range(0, len(latencies)) :
            latency = latencies[i]
            destination = destinations[i]

            ## Setting up class for specific bandwidth limiting
            os.system(f"tc class add dev {dev} parent 1:0 classid 1:{class_idx} htb rate {bandwidth}mbps ceil {bandwidth}mbps")

            ## Setting up latency using netem
            os.system(f"tc qdisc add dev {dev} parent 1:{class_idx} handle {class_idx}: netem delay {latency}ms 1ms distribution normal")

            ## Setting up filter for specific destination
            os.system(f"tc filter add dev {dev} parent 1:0 protocol ip prio 1 u32 match ip dst {destination} flowid 1:{class_idx}")

            class_idx += 1

    return



if __name__ == "__main__" :
    main()