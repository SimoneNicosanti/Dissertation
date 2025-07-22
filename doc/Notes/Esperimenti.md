
## Test Tempo di Inferenza (Preliminare per Prof)
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

# SetUp degli Esperimenti

## Modelli da Analizzare
Possiamo fare degli esperimenti preliminari su tutte le combinazioni: ad esempio, considerare una tabella in cui si vede il numero totale di nodi ed archi per capire la dimensionalità del problema di ottimizzazione.

TABELLA CON: Numero di Nodi e Numero di Archi (per vedere la dimensionalità)

Ci possiamo restringere a: task di classicazione e task di segmentazione considerando un sottoinsieme delle dimensioni (ad esempio modelli small e x) in funzione del numero di nodi e di archi che sono coinvolti nel problema.


## SetUp Server e Rete


> [!NOTE] 
> Questo set up dei nodi e della rete deve essere quanto più possibile fisso: possiamo fare delle piccole modifiche per indurre alla divisione del modello, ma se ad un certo punto fa offloading totale va bene anche così, evidentemente è più conveniente.


Caratteristiche Device:
- CPU: 0.5
- Supporto OpenVINO: yes

Caratteristiche Edge:
- CPU: 1 oppure NO LIMIT
- Supporto OpenVINO: yes
- Supporto GPU: no

Caratteristiche Cloud:
- CPU: no limit
- Supporto OpenVINO: no
- Supporto GPU: yes

ATTENZIONE: Fissate le caratteristiche dei dispositivi si può intervenire a livello di banda di rete (oppure fissare le bande e le latenze, modificando le capacità computazionali dei server per avere delle buone divisioni).

Bande di Trasmissione:
- Assumi 4G/5G device-edge: vedere [[Note Sugli Articoli#Modellazione del consumo energetico di trasmissione]]
- Assumi Eth edge-cloud (100 circa)

Latenze di Trasmissione:
- 5 ms : device-->edge
- 50 ms : edge --> cloud
- 55 ms : device --> cloud
Scenario best-station: il device si connette ad una best station da cui passa anche la connessione verso il cloud, quindi possiamo supporre linearità nella latenza.

Parametri di Consumo:
- Calcolo
	- Consumi nominali in base ad architetture
	- Stiamo assumendo OpenVINO e GPU, quindi bisogna vedere tendenzialmente questi consumi
- Trasmissione ad Altri: Vedere [[Note Sugli Articoli#Modellazione del consumo energetico di trasmissione]]
- Trasmissione a se stesso
	- Assumiamo trascurabile rispetto al consumo dato dalla trasmissione ad altri, quindi mettiamo coefficiente e costante a 0

## Scenari da Considerare

Parametri del Piano:
- Pesi
	- 5 Casi principali
		- (1,0)
		- (0.75, 0.25)
		- (0.5, 0.5)
		- (0.25, 0.75)
		- (0, 1)
- Massima Energia Device
	- Possiamo fare un aumento progressivo
		- Caso 1: 0 --> Non Considerata
		- Caso 1, 2, 3 --> Aumentata mano mano, anche in base ai consumi energetici di trasmissione e calcolo
- Rumore di Quantizzazione:
	- Aumento progressivo:
		- Caso 1: 0
		- Caso 2: 0.1
		- Caso 3: 0.2
		- Caso 4: 10

Struttura del modello:
- Non quantizzato: con quantizzazione attivata tramite regressore
- TUTTO Quantizzato: con quantizzazione fatta offline e regressore messo a 0 (VEDERE SE SI PUò FARE EFFETTIVAMENTE O SE NON CREA PROBLEMI).


# Preliminari

## Numero di Nodi ed Archi per Modello

Di seguito il numero di nodi e di archi del modello al variare del task e della dimensione: 
- I modelli di classificazione hanno un numero di nodi molto basso: il caso x ha un numero di nodi più basso del caso n di segmentazione/detection. Volendo si potrebbe usare direttamente il tipo x per questo modello, senza perdere troppo tempo con il caso più piccolo
	- Potremmo considerare. Queste combinazioni coprirebbero varie dimensioni e varie classi di modello:
		- n-cls come il caso più piccolo di modello
		- m come come intermedio alto
		- x-seg come caso massimo (poco più piccolo dell' x-seg)
- Nota: ci sono alcuni modelli che sebbene più grandi hanno lo stesso numero di nodi. Questo è probabilmente dato dal fatto che la differenza sta nel numero di parametri (e.g. dimensione del kernel dei livelli convoluzionali).
![[Schermata del 2025-07-02 17-07-18.png]]

Restringiamo l'analisi a tre dimensioni, una per ogni task:
- yolo11n-cls (il più piccolo) per il task di classificazione
- yolo11m (dimensione media in numero di nodi) per il task di detection
- yolo11x-seg (dimensione massima) per il task di segmentazione


## Profiling del Modello
Tra le 5 e 10 run

Parametri di Test:
- Macchina
	- GPU Tesla T4
	- n1-standard-8
- Runs
	- 5
- Quantizzazione
	- Calibration set size: 100
	- Noise Test Set Size: 20
	- Max Quantizable: 12
- Regressore
	- Max Degree: 3
	- Train Set Size: 1000
	- Test Set Size: 100

Note Aggiuntiva:
- Nel caso di Yolo11n-cls ci sono due livelli Sigmoid che sono candidati alla quantizzazione. Specificare questa cosa nella discussione sul profiling perché detto il contrario nel capitolo dui dettagli implementativi.
- Specificare che la maggior parte del carico in termini di profiling è dato da
	- Molte versioni quantizzate che vengono create
	- Run di Test per il rumore
		- Per attenuare il passaggio dei dati in memoria, il dataset di test viene caricato in GPU una volta sola attraverso API OnnxRuntime e gli OrtValues
		- L'output deve essere sempre copiato dal device per il confronto



## Profiling del Server

Configurazioni possibili per cui fare profiling sui server:

|          | Device         | Edge              | Cloud  |
| -------- | -------------- | ----------------- | ------ |
| CPUs     | 0.5 / 0.75 / 1 | 1 / 1.25 / No Lim | No Lim |
| OpenVINO | yes            | yes               | /      |
| CUDA     | No             | No                | Yes    |

Questi sono i casi di interesse principali: sono comunque indipendenti tra di loro.

Parametri di Test:
- Generali
	- Runs
		- Per Livello: 5
		- Per Profiling Totale: 5



Device:
- GCP
	- c3-standard-4
	- Supporto VNNI
- CPUs
	- 0.5
	- 0.75
	- 1

Edge:
- GCP
	- c3-standard-4
	- Supporto VNNI
- CPUs
	- 1.0
	- 1.25
	- No Limit

Cloud:
- GCP
	- n1-standard-4
	- Supporto GPU
- CPUs
	- No Limit
- GPU
	- Tesla T4

Annotazioni Aggiuntive:
- Per limitare l'impatto del tempo di spostamento dei dati nel profiling del Cloud in caso di GPU, si prealloca il tensore di input sulla GPU e poi si eseguono le run, in modo da non considerare il tempo di trasferimento (che altrimenti verrebbe anche considerato più di una volta)
- Trattandosi delle stesse macchine, il caso di CPU 1 per Device ed Edge sono lo stesso caso
	- Posso evitare di eseguire 2 volte la stessa cosa (cambia i nomi dei file)
- Modificato profiling OpenVINO
	- Fatto con le API di OpenVINO e non con quelle di OnnxRuntime
		- Veniva aggiunto un overhead eccessivo

## Esecuzione sul Device

Fatta esecuzione dei modelli target sul device con le configurazioni di CPUs volute.

Confrontato con tempo di profiling ottenuto in fase di server profiling. Il confronto per adesso è fatto soltanto rispetto al device e alle sue configurazioni di CPU. Non sto facendo confronti sul profile d GPU (anche perché non riesco ad istanziare la macchina cloud).


## Generazione del Piano

PER ADESSO TRASCURIAMO LA GENERAZIONE DEL PROBLEMA CON LIMITE DI MEMORIA! MODIFICARE LA FORMULAZIONE DELLA MEMORIA NEL PROBLEMA!

### Parametri Energetici

#### Calcolo
Consideriamo il consumo energetico dell'hardware su cui mandiamo in esecuzione:
- Cloud
	- GPU T4
		- Potenza massima nominale 70 W
		- https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
- Edge / Device
	- c3-standard-4 --> Processore Intel Xeon di 4ª generazione
	- https://www.intel.com/content/www/us/en/developer/articles/technical/fourth-generation-xeon-scalable-family-overview.html
		- Eventuale rescaling del TDP. Avremmo ad esempio:
		   $$\frac{350 \hspace{0.2cm} W}{60 \hspace{0.2cm} core} \cdot docker\_cpus = 5.833 \cdot docker\_cpus$$
		   Nel caso di 0.5 avremmo circa 2.916, con 0.75 4.375.
		   Quando abbiamo il NO_LIMIT abbiamo le cpus=4, quindi 23.33 circa.


#### Trasmissione
[[Note Sugli Articoli#Modellazione del consumo energetico di trasmissione]]

|                        | 4G      | Wi-Fi  | Eth (Cloud) |
| ---------------------- | ------- | ------ | ----------- |
| Upload Speed (Mbps)    | 5.85    | 18.88  | 800         |
| $\alpha_u$ (mW / Mbps) | 438.39  | 283.17 |             |
| $\beta$ (mW)           | 1288.04 | 132.86 |             |

La formula per il consumo è $P_u = \alpha_u * t_u + \beta$, dove $t_u$ è il thr (che noi misuriamo con iperf3, quindi va bene).

Tabella convertita:
CONTROLLARE i Valori per CLOUD

|                       | 4G      | Wi-Fi   | Eth (Cloud) |
| --------------------- | ------- | ------- | ----------- |
| Upload Speed (MB/s)   | 0.73125 | 2.36    | 100         |
| $\alpha_u$ (W / MB/s) | 3.5071  | 2.2654  | 0.014       |
| $\beta$ (W)           | 1.28804 | 0.13286 | 0.5         |


## Caso 1 - Solo Device
In questo caso quello che posso studiare è la latenza aggiuntiva che l'uso del sistema introduce in fase di inferenza.

Analisi:
- Tempo di Generazione del Piano 
	- Mediato su una decina di run
- Tempo di Deployment del Piano - Potrebbe essere non banale per estrazione di modelli grandi e quantizzazione
	- Mediato su una decina di run
- Esecuzione del Modello
	- Mediato su 100 Run

Casi da considerare:
- Rumore Quantizzazione
	- 0; 0.05; 0.1; 0.25; 0.5; 10
- Pesi di Latenza ed Energia
	- Quantizzazione = 0 --> Non cambia nulla
	- Quantizzazione != 0 --> Può cambiare
	- Casi da Considerare per latenza 
		- 1, 0.75, 0.5, 0.25, 0

Annota e salva i valori di output del piano!


## Limiti di Container

Processori delle macchine (https://cloud.google.com/compute/docs/cpu-platforms?hl=it):
- C3-Standard
	- Sapphire Rapids
	- Freq Base >> 2.2
- E2-Standard
	- Skylake
	- Freq Base >> 2.0

Le differenze potrebbero essere dovuto ai processori diversi??

| Macchina      | Numero CPUs | Modello | Tempo Medio [s] | Avg Predetto [s] |
| ------------- | ----------- | ------- | --------------- | ---------------- |
| c3-standard-4 | 1           | x-seg   | 2.2             |                  |
| c3-standard-4 | 1           | m-det   | 0.55            |                  |
| c3-standard-4 | 2           | m-det   | 0.27            |                  |
| c3-standard-4 | 2           | x-seg   | 1.11            |                  |
| c3-standard-4 | 3           | m-det   | 0.28            |                  |
| c3-standard-4 | 3           | x-seg   | 1.11            |                  |
|               |             |         |                 |                  |
| e2-standard-4 | 1           | x-seg   | 5.2             | 6.12             |
| e2-standard-4 | 1           | m-det   | 1.26            | 1.95             |
| e2-standard-4 | 2           | x-seg   | 2.64            |                  |
| e2-standard-4 | 2           | m-det   | 0.63            |                  |
| e2-standard-4 | 3           | x-seg   | 2.63            |                  |
| e2-standard-4 | 3           | m-det   | 0.63            |                  |

La differenza può sicuramente essere imputata alle ottimizzazioni aggiuntive fatte dal provider.


Yolo11x-seg profiling
![[Schermata del 2025-07-21 19-14-06.png]]

Yolo11m profiling
![[Schermata del 2025-07-21 19-20-28.png]]
NOTA Aggiuntiva: con il modello yolo11m quantizzato, il tempo medio di inferenza per il modello completo con affinity=0 (singola CPU) è circa 0.78, quindi anche qui la predizione può essere abbastanza precisa.
![[device_predicted_vs_real.png]]


Fatto anche il confronto con il profiling livello per livello su edge: più o meno ci siamo, ma ci sono circa 700 ms di ottimizzazioni che non vengono catturati in modo opportuno.

Per GPU invece ci siamo: è stato necessario fare run con IOBinding, ma tutto sommato nulla di problematico

## Scalabilità Problema
Provare partendo da un modello Yolo, crea un modello fittizio per vedere la scalabilità del problema.