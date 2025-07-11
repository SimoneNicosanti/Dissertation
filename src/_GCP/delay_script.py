import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--dev", type=str, help="Interface Name", required=True)
    parser.add_argument(
        "--bandwidth", type=float, help="Bandwidth [MBps]", default=10_000
    )
    parser.add_argument("--latencies", nargs="+", type=float, help="Latencies To Add")
    parser.add_argument("--ips", nargs="+", type=str, help="Destination ips")
    args = parser.parse_args()

    dev = args.dev
    bandwidth = args.bandwidth
    destinations = args.ips
    latencies = args.latencies

    # Clearing up root qdisc
    # trunk-ignore(bandit/B602)
    subprocess.run(f"tc qdisc del dev {dev} root || true", shell=True, check=False)

    # Creating default class for unclassified packets
    subprocess.run(
        f"tc qdisc add dev {dev} root handle 1:0 htb default 1",
        # trunk-ignore(bandit/B602)
        shell=True,
        check=False,
    )

    # Default class with bandwidth limit
    subprocess.run(
        f"tc class add dev {dev} parent 1:0 classid 1:1 htb rate {bandwidth}mbps ceil {bandwidth}mbps",
        # trunk-ignore(bandit/B602)
        shell=True,
        check=True,
    )

    if latencies is not None and destinations is not None:
        if len(latencies) != len(destinations):
            raise ValueError("Latencies and Destinations must have the same length")

        class_idx = 10
        for latency, destination in zip(latencies, destinations):
            # Setting up class for specific bandwidth limiting
            subprocess.run(
                f"tc class add dev {dev} parent 1:0 classid 1:{class_idx} htb rate {bandwidth}mbps ceil {bandwidth}mbps",
                # trunk-ignore(bandit/B602)
                shell=True,
                check=True,
            )

            # Setting up latency using netem
            subprocess.run(
                f"tc qdisc add dev {dev} parent 1:{class_idx} handle {class_idx}: netem delay {latency}ms 1ms distribution normal",
                # trunk-ignore(bandit/B602)
                shell=True,
                check=True,
            )

            # Setting up filter for specific destination
            subprocess.run(
                f"tc filter add dev {dev} parent 1:0 protocol ip prio 1 u32 match ip dst {destination} flowid 1:{class_idx}",
                # trunk-ignore(bandit/B602)
                shell=True,
                check=True,
            )

            class_idx += 1


if __name__ == "__main__":
    main()
