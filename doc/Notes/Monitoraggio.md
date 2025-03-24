# Memoria

La memoria si può monitorare usando *psutil* con il metodo *virtual_memory*
`psutil.virtual_memory().available` che ritorna la memoria disponibile nel sistema in bytes.

Il problema è quando si esegue il server nel container per limitare le risorse; pur usando il flag --memory di docker, comunque psutil individua la memoria dell'host e non quella disponibile al container. Il motivo è che psutil si basa per l'estrazione delle statistiche su /proc/meminfo, che sono impostate a quelle del sistema; docker invece gestisce il limite della memoria usando *cgroup*. 

Nella directory */sys/fs/cgroup/* troviamo i due file:
- memory.current
- memory.max
Dove il secondo è proprio il file in cui viene scritto il limite di memoria dato al container

https://facebookmicrosites.github.io/cgroup2/docs/memory-controller.html
https://www.kernel.org/doc/Documentation/cgroup-v2.txt

Quindi se questi due file esistono, la memoria disponibile è calcolata come `max - current`, altrimenti si prende la statistica come letta da psutil.

# Banda / Latenza
La banda si può vedere per tipo di macchina su https://cloud.google.com/compute/docs/network-bandwidth?hl=it. In particolare alla sezione https://cloud.google.com/compute/docs/network-bandwidth?hl=it#vm-out-baseline si trovano i limiti della banda per tipo di macchina. 

La regola è che sono forniti 2 Gbps per vCPU.

Nella sezione https://cloud.google.com/compute/docs/general-purpose-machines?hl=it si trovano le specifiche per coppia (tipo di macchina, tipo di istanza).

Da qui quindi si può risalire alla banda che la macchina ha a disposizione.

Per la latenza invece quella si può calcolare tranquillamente usando il ping e modellarla a seconda delle necessità usando tc.

Anche la banda la posso limitare con tc!!
`tc class add dev $DEV parent 1: classid 1:15 htb rate 100000Mbps quantum 1500`
In questo caso (credo) che il rate rappresenti proprio questo, il tasso di trasmissione di pacchetti di quella categoria

## Configurazioni di Rete
Le configurazioni di rete da analizzare sono:
1. Banda da x --> y
2. Latenza da x --> y
Questo per ogni coppia (x,y) di dispositivi.

La banda e la latenza le imposto usando tc

# Flops
Un modo indiretto per la regolazione dei flops è l'uso del flag --cpus di docker. Usando questo si può limitare il numero di cpu a cui il container ha accesso e quindi in sostanza il totale di operazioni che il server può fare nell'unità di tempo

# Consumo

## Calcolo

## Trasmisssione