
Punto della situazione:
- Device deve/può
	- Eseguire parte della computazione
		- In funzione dello stato energetico
		- Questa parte può essere la prima (feature extraction o una sua sotto parte) o l'ultima ad esempio
	- Scegliere un modello più o meno accurato
		- In funzione di
			- Condizione della rete
			- Stato della batteria
	- Scegliere una compressione dei dati
		- Per diminuire il tempo di invio (a costo di minore accuratezza nella risposta)
		- La compressione dei dati può essere fatta anche considerando un punto in cui la rete si "assottiglia", cioè i dati che sono elaborati dalla rete diventano meno
- Il modello deve/può essere partizionato per supportare questi aspetti
	- La divisione può essere fatta in funzione della capacità di calcolo/memoria del dispositivo a cui quella parte di modello viene assegnata
	- Un singolo server può anche gestire più parti contemporaneamente e/o più modelli e/o più varianti dello stesso modello (ad esempio quantizzazione)





Possibile flusso.
- Model Divider
	- Quando il modello è caricato, Model Divider si occupa di dividerlo in base a 
		- Capacità server edge/cloud
			- FLOPS che il possibile sotto modello richiede
	- Una volta diviso i sotto modelli sono salvati in Model Pool
- Model Pool
	- Mantiene i modelli completi e divisi
	- I server e i Device possono accedergli per capire prendere la loro parte del modello da operare
- Model Servers
	- Gestiscono dei sotto modelli o dei modelli completi
	- Accolgono le richieste ed elaborano gli output
	- Espongono delle API di detection dello stato/risorse
		- Esempio
			- Carico di lavoro attualmente in gestione
			- Modelli attualmente gestiti
			- Memoria/Capacità di calcolo
		- Interrogato da
			- Divider per capire come dividere
			- Planner per capire il piano di lavoro per una richiesta
- Front-End Server
	- Componente di disaccoppiamento tra Device e Server
	- Fa le richieste delle parti di lavoro ai model server per conto del device
- Execution Planner
	- Noti i requisiti di una richiesta e le capacità di calcolo, stabilisce il piano di esecuzione della richiesta
		- Latenza Massima
		- Stato energetico del device
		- Capacità calcolo del device
		- Banda disponibile
		- ecc.
- Registry
	- Traccia Model Servers disponibili
- Remote Detection Storage
	- Storage dove vengono mantenuti i risultati del task in questione


> [!Warning] Divisione e conseguenze su Device
> I device che sono nel sistema potrebbero essere diversi e avere capacità di esecuzione diversi. Una divisione che va bene per un device potrebbe non andare bene per un altro.
> 
> Allo stesso tempo non è detto che un device faccia una richiesta, quindi se faccio la divisione in base a tutti i device potrei tenere in conto alcuni non attivi in cerrti momenti (e.g. non attivi).
> 
> Potrei magari pensare ad una divisione adattativa?? Tengo in conto nella divisione solo quei device che sono più attivi a mandare le richieste.

> [!Warning] Organizzazione dei Server
> Non posso organizzare i server ad anello stile Chord: potrebbero esserci più modelli che hanno bisogno dello stesso tensore di output. In sostanza potrei avere più di un nodo successore.
> Ammesso di organizzare i successori come quelli che devono ricevere l'output del sotto modello corrente, possono essercene di più.
> 
> Un'alternativa a questo potrebbe essere quella di avere un anello di balancer che bilanciano le richieste di più modelli che gestiscono le stesse parti di sotto modelli.
> 
> Oppure dividere il modello in modo che questa cosa non si verifichi.

> [!Tip] Planner e Divider
> Approcci per la pianificazione/divisione possono essere:
> - Programmazione Lineare Intera
> - Reinforcement Learning (generazione di scenari simulati)
> - Euristica


---


Device avvia richieste di inferenza ai server (ok).

Client richiede deployment di un modello sull'infrastruttura.

Architettura di Sistema con NODI che compongono il sistema. Nei nodi ho i componenti. 

DATA Plane vs CONTROL Plane

DATA Plane --> Nodi 
CONTROL Plane --> Componenti che sono collocati sull'infrastruttura (Parliamo in termini di Ruoli)

Decidi come dividere la computazione in base a come allochi la divisione (tenendo conto che i modelli sono pesanti).

Considera un Optimizer: fa sia la divisione sia il piano di computazione.
Fusione degli aspetti: come divido e dove piazzo da considerare insieme. 

> [!NOTE]  Aspetto Secondario
> L'Optimizer crea più varianti dello stesso modello:
> - modello suddiviso 1: prima parte in modello e resto in cloud
> - modello suddiviso 2 : prima parte in edge e resto in cloud
> Logica che smista tra queste varianti

Profiler per il modello (ok)
Monitor per Latenza di rete/capacità di calcolo.

Fai dei sequence diagram per ragionare meglio sulle interazioni.

Comincia a ragionare sulle politiche di split.

---

## Model Deployment

![[ModelDeployment.svg|Model Deployment con Pull dello Stato]]
Breve descrizione delle interazioni:
1. **Client** richiede il deployment di un modello sul sistema
2. Il **Deployer** contatta la componente di ottimizzazione per ottimizzare il modello richiesto
3. **Optimizer** procede in questo modo
	1. Contatta **Registry** per ottenere la lista dei **server** attualmente attivi
	2. Per ogni **Server** viene contatto un **ServerMonitor** che restituisce informazioni riguardo il server; queste informazioni possono essere del tipo
		1. Banda disponibile e/o metrica di vicinanza agli altri server (più sono "vicino" meno ci metto ad inviare i dati; potrei considerare ad esempio il tempo medio di ping agli altri nodi del sistema)
		2. Memoria RAM residua
		3. Capacità di calcolo
	3. Note queste informazioni, viene fatta l'ottimizzazione del modello: vengono prodotti diversi piani di partizionamento del modello in cui ad ogni parte del modello diviso viene assegnato un server. Qui possiamo tenere in considerazione gli aspetti di:
		1. Quantizzazione
		2. Varianti di modelli (esempio, YOLO)
	4. Le varie parti prodotte vengono salvate nel **ModelPool** assieme a dei metadati di identificazione della versione del modello
	5. Ordina un pull ai diversi **Server** che mandano in servizio quella parte del modello
		1. Il **Client** che avvia l'inferenza, essendo in grado di eseguire il modello è da considerarsi al pari di qualsiasi altro server.

In questo caso abbiamo un modello pull dello stato: il deployer chiede lo stato ad ogni server che il registry dice essere attivo. 
In alternativa si potrebbe fare un modello push: ogni server periodicamente invia il suo stato ad un componente di **StatePool** che raccoglie gli stati di tutti i server del sistema. Avrei una componente centralizzata in più, però è più veloce la raccolta dello stato (soprattutto considerando casi di riconfigurazione della rete).

![[Deployment_Push.svg|Model Deployment con Push dello Stato]]


## Inferenza

![[Inference.svg|Inferenza]]
Descrizione delle interazioni:
1. **ClientFrontEnd** è la componente del client che riceve le richieste di inferenza; permette di disaccoppiare il sistema di raccolta delle immagini dall'inferenza sulle stesse
	1. Raccoglie lo stato energetico del dispositivo dal **Monitor** del dispositivo stesso
	2. Invia all'**Optimizer** una richiesta di produzione di piano per l'inferenza
	3. Ricevuto il piano ottimizzato esegue le richieste ai **Server**
		1. Anche qui avremo un'istanza di **Server** in esecuzione nel **Client**
	4. Il risultato viene tornato attraverso un callback da cui poi si salva il risultato, qualora il suo salvataggio sia necessario.
		1. In caso di fallimento, la callback potrebbe avviare di nuovo l'inferenza (posto che possa risalire all'input a cui corrisponde)
2. L'**Optimizer** può ottimizzare il piano. Possiamo considerare tre aspetti.
	1. L'ottimizzazione può essere fatta sulla base di informazioni come:
		1. Latenza massima voluta
		2. Accuratezza minima voluta (o misura correlata)
		3. Consumo energetico massimo / Livello di Consumo (assumendo una divisione in fasce di consumo) (questo aspetto in realtà è anche in parte legato all'accuratezza, quindi uno dei due si potrebbe incorporare nell'altro ??)
	2. La parte di modello in esecuzione sul **Client** dovrebbe essere forzata ad essere la prima e/o l'ultima, in modo che vi sia compressione dei dati nella prima fase e veloce ricezione dell'output nell'ultima
	3. Volendo, si potrebbero produrre piani alternativi: fissata una divisione del modello, se una sotto parte viene mandata in esecuzione su più server, si potrebbero produrre più piani per avere tolleranza ai guasti.

Assumiamo che l'Optimizer abbia già prodotto dei piani e in funzione dei requisiti di energia, latenza e accuratezza della richiesta venga scelto uno di questi piani (assumiamo che ci sia stato una sorta di pull di una tabella di LookUp).



## Visione Generale
![[General.svg|Visione Generale - Prima con pull e poi con push|650]]


## Piani
Esempio di Piani:
- Plan_0: esecuzione locale
- Plan_1: l'output del server_0 può essere mandato sia a server_1 che a server_2, per poi continuare. Questo potrebbe essere ad esempio per tolleranza ai guasti: se non risponde 1 lo mando a 2
- Plan_2: l'output di server_0 viene mandato in parte a server_1 e in parte a server_2 per poi ricongiungere il tutto su server_3
![[Plans.svg|Generazione di Piani Diversi|650]]




# Implementazione

Da qui si vede che il GIL non è un problema con la *run* di ONNX: infatti dice esplicitamente che il GIL viene rilasciato per permettere la chiamata di *run* da parte di più thread
https://github.com/microsoft/onnxruntime/issues/11246.
Di base quindi si potrebbero eseguire più modelli senza creare dei processi diversi.
Ad esempio si potrebbe creare un wrap del modello e delle sue componenti che espone un metodo di Run: il metodo di run internamente gestisce un semaforo per limitare il numero di thread che contemporaneamente possono andare a chiamare quel servizio.