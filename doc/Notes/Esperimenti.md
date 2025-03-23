- Temporizzazione per l'inferenza

Simulazione ritardo di rete: traffic control (TC)

Latenze:
device --> edge 5 ms
device --> cloud circa 80 ms

Scenari:
device --> edge --> cloud

Vedere script per i ritardi di rete (di base lo script è per ritardare pack a specifico ip).
Per ritardo pack su certa interfaccia vedi online

Prendere scheda dell'edge (NVidia Jatson o raspberry); vedere quanto consuma in caso full e moltiplicare.

(Trascurare Consumo Energetico e fare solo su latenza).

CGroup (Quota della CPU)
Container e docker:
- --cpus per limitare il tempo di cpu
- --memory per limitare la memoria

Vedere se nel problema c'è la latenza modellata (se non c'è aggiungere termine additivo)

Con il TC puoi configurare sia la latenza sia la banda di tx.

Cominciare da parte centrale...



## Banda e Latenza
La banda si può vedere per tipo di macchina su https://cloud.google.com/compute/docs/network-bandwidth?hl=it. In particolare alla sezione https://cloud.google.com/compute/docs/network-bandwidth?hl=it#vm-out-baseline si trovano i limiti della banda per tipo di macchina. 

La regola è che sono forniti 2 Gbps per vCPU.

Nella sezione https://cloud.google.com/compute/docs/general-purpose-machines?hl=it si trovano le specifiche per coppia (tipo di macchina, tipo di istanza).

Da qui quindi si può risalire alla banda che la macchina ha a disposizione.

Per la latenza invece quella si può calcolare tranquillamente usando il ping e modellarla a seconda delle necessità usando tc.

Anche la banda la posso limitare con tc!!
`tc class add dev $DEV parent 1: classid 1:15 htb rate 100000Mbps quantum 1500`
In questo caso (credo) che il rate rappresenti proprio questo, il tasso di trasmissione di pacchetti di quella categoria



# Configurazioni di Rete
Le configurazioni di rete da analizzare sono:
1. Banda da x --> y
2. Latenza da x --> y
Questo per ogni coppia (x,y) di dispositivi.

La banda e la latenza le imposto usando lo script di tc


# Configurazione delle Risorse
Per ogni dispositivo possiamo considerare:
- Memoria
- FLOPS
Entrambi si possono configurare usando docker con --cpu e --memory.

Altrimenti si possono configurare usando delle macchine diverse.