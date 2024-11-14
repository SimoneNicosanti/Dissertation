import socket
import psutil
import matplotlib.pyplot as plt
import time
import numpy as np
import os

# Parametri del client
HOST = '127.0.0.1'  # Indirizzo del server
PORT = 65432        # Porta del server

# Nome del file da inviare
file_path = 'zeros.txt'

def main() :
    fig = plt.figure(figsize = (6, 9))
    ax = plt.subplot()

    batteryList = []
    timeList = []
    for _ in range(0, 100) :
        start = time.time()
        batteryList.append(psutil.sensors_battery().percent)
        # Creazione del socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print("Connesso al server su", HOST, ":", PORT)
            
            # Lettura del file e invio dati
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(1024), b''):
                    s.sendall(chunk)  # Invia pacchetti di 1024 byte
            print("File inviato con successo")
        end = time.time()
        timeList.append(end - start)
    
    batteryList = np.array(batteryList) - batteryList[0] + 100
    ax.plot(batteryList, label = "Network")

    batteryList = []
    for x in range(0, 100) :
        batteryList.append(psutil.sensors_battery().percent)
        time.sleep(timeList[x]) 
        print("Iteration >>> ", x)
    batteryList = np.array(batteryList) - batteryList[0] + 100
    ax.plot(batteryList, label = "Idle")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Battery %")
    ax.legend()
    fig.suptitle("Network Consumption")
    fig.tight_layout()
    fig.savefig("./NetworkConsumption")

    print("Avg Send Time >>> " + str(np.array(timeList).mean()))
        

if __name__ == "__main__" :
    main()
