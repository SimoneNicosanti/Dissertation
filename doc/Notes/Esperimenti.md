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
