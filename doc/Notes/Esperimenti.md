
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
- Supporto GPU: yes

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

