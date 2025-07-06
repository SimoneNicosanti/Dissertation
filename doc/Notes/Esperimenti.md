
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



## Scalabilità Problema
Provare partendo da un modello yolo, crea un modello fittizio per vedere la scalabilità del problema.