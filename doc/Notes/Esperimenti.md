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







## Test Tempo di Inferenza
Qui è abbastanza facile perché dipende dal piano che è stato generato.

Piccolo Esempio:
- Yolo11l
- Massima Quantizzazione Possibile
- 100 Run
- Provider OpenVINO
- Server UNICO (device)
![[Schermata del 2025-06-19 13-54-29.png]]

Il risultato del problema di ottimizzazione è il seguente
![[Schermata del 2025-06-19 15-01-30.png]]

Notiamo che non c'è corrispondenza in termini di valore tra il predetto ed il valore effettivo, però le differenze sembrano già più vicine:
- Sui valori reali abbiamo 0.17007
- Sui valori teorici abbiamo 0.2516866
Bisogna un attimo vedere questa cosa meglio, perché ci possono essere molti aspetti da tenere in considerazione.

