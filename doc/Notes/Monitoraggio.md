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


### Banda verso se stesso
è importante che la banda da un dispositivo a se stesso sia valutata in modo accurato. Non posso mettere la stessa banda del dispositivo verso un altro perché non è realistico!! Inoltre l'ottimizzatore non riesce a dividere in queste situazioni.

### Latenza
Calcolata con il ping (e ok)

### Banda di rete
Non so come stimarla... o considero il thr assumendo che la latenza sia trascurabile, oppure considero la banda di rete come data dalle specifiche delle schede / della macchina.

La banda viene monitorata usando iperf3 con pacchetti UDP.
Il monitor di un server fa partire un server iperf3; le connessioni vengono ricevute sia dal monitor del server stesso (banda da x a se stesso), sia da altri monitor (effettiva banda di rete).

Iperf3 sembra essere single thread! (https://github.com/esnet/iperf/issues/980) Quindi mi attacco dove so io!! Da qui vengono gli errori degli screen successivi:
![[Schermata del 2025-03-25 17-41-30.png]]
![[Schermata del 2025-03-25 17-41-42.png]]

A questo punto tocca usare iperf2! Oppure semplicemente zompo il check della memoria

Dalla versione 3.16, come indicato qui, iperf3 è diventato multithread https://fasterdata.es.net/performance-testing/network-troubleshooting-tools/iperf/, quindi bisogna impostare bene la versione!

## Limite sulla banda di rete
Il limite sulla banda di rete viene impostato come spiegato in https://netbeez.net/blog/how-to-use-the-linux-traffic-control/

Il burst viene impostato come indicato da (LeMistral) (Cercare controprova di questa cosa online)
![[Schermata del 2025-03-25 16-49-52.png]]

Comunque impostare il burst in questo modo permette di monitorare una banda limitata come quella indicata nel limite sullo script.

Usare tbf non sembra funzionare come dovrebbe.

Seguento https://robert.nowotniak.com/security/htb/ e usando htb si riesce a configurare in modo opportuno: sembra che il trucco fosse la creazione di una classe intermedia genitore. Il problema è che quando uso netem per configurare la latenza, il limite sulla banda non viene più imposto! 

Dalla documentazione di netem https://man7.org/linux/man-pages/man8/tc-netem.8.html
sezione Limitations:
> [!Quote] Limitations
> Combining netem with other qdisc is possible but may not always work because netem use skb control block to set delays.

Quindi bisogna fare attenzione a come il ritardo di rete viene simulato!


In realtà facendo delle prove si vede che HTB la banda la limita comunque anche senza la seconda classe! Il problema è solo netem che scombina tutto quanto e fa cose strane rompendo il tutto.

In questo link, viene riportato che netem invia tutti quanti i dati immediatamente, senza aspettare.
https://www.linuxquestions.org/questions/linux-networking-3/netem-the-network-emulator-issue-4175500681/

In questo link viene suggerito l'uso di HFSC anziché di HTB 
https://serverfault.com/questions/583788/implementing-htb-netem-and-tbf-traffic-control-simultaneously

In questo link viene riportato un bug nel dettaglio (è del 2017, però può essere indicativo) https://patchwork.ozlabs.org/project/netdev/patch/20170313171658.18606-1-sthemmin@microsoft.com/


### Conclusione
Prima il monitoraggio della banda era fatto cercando di considerare la banda teorica, vedendo quindi quanto traffico UDP si potesse far passare in un certo tempo attraverso l'interfaccia di rete. Ragionando un attimo credo che la cosa migliore sia fare la misura usando TCP; è vero che TCP è influenzato dalla latenza che si imposta sul link, ma è altrettanto vero che è quello che bisogna considerare:
- il sent_MB_s mi dice quanti MB di dati posso inviare al secondo usando TCP nelle condizioni attuali di rete
- Visto che gRPC usa TCP considerare la banda calcolata con UDP lascia il tempo che trova e potrebbe essere deleterio in certa misura visto e considerato che comunque quella velocità teorica NON la posso raggiungere tenendo conto di tutto l'overhead dato da TCP

In questo modo, quando imposto la latenza la banda possibile si abbassa inevitabilmente per via dell'attesa dell'ack da parte di TCP. Anche quando non viene introdotta latenza aggiuntiva, comunque la banda che trovo è minore di quella che imposto, sempre per Ack e tutte cose.

Quando alzata la latenza all'improvviso ci sono due fasi:
1. In un primo momento la banda diminuisce tutta insieme
2. Poi si alza e si stabilizza
La cosa è probabilmente dovuta al fatto che quando la latenza viene alzata, TCP si deve un attimo riconfigurare con le finestre: bisogna fare attenzione a questo fatto e aspettare un po' prima di valutare il comportamento del sistema.


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