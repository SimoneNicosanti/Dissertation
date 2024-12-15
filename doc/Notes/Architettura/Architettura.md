
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



![[Diagram_1.svg]]

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

