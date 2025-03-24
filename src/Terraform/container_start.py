import os
import argparse
def main() :

    parser = argparse.ArgumentParser()

    # Aggiungi argomenti
    parser.add_argument("--memory", type=float, help="Memory Size for Container")
    parser.add_argument("--cpus", type=float, help="CPUs percentage for Container")

    # Parse degli argomenti
    args = parser.parse_args()

    os.system("docker stop server-container")
    os.system("docker remove server-container")
    os.system("docker build -t server-image -f server.dockerfile .")
    
    command = "docker run -it -d --name server-container -v ~/src/:/src/ --network host"
    if args.memory is not None :
        command += f" -m {args.memory}g"
    if args.cpus is not None :
        command += f" --cpus={args.cpus}"
    command += " server-image"
    print(command)
    os.system(command)

    os.system("docker exec -it server-container /bin/bash")



if __name__ == "__main__" :
    main()