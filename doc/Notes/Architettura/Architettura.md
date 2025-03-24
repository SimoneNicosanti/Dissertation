
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


## Aggiornamento dell'Architettura
![[Deployment_Diagram_2.svg]]
Rispetto alla versione precedente, la profilazione del modello può essere fatta anche una sola volta! E soprattutto viene fatta in fase di upload del modello. In questo caso il ruolo del ModelManager è quello di profilare e dividere il modello, così come quello di inviare le componenti quando richieste. In sostanza quindi unisce il ruolo di profiler e di pool. Infatti il problema è che sia per fare il profiling sia per dividere il modello, questo modello bisogna averlo. A questo punto può risultare più conveniente avere un servizio che si occupi di entrambi questi aspetti piuttosto che avere più servizi che lo fanno: in questo modo si evita di trasferire in lungo e in largo il modello, altrimenti il modello deve essere:
1. Caricato nel pool
2. Spostato per fare il profiling
3. Spostato per fare la divisione
4. Spostate le componenti
Con il ModelManager i punti 2 e 3 si possono evitare.

![[Architecture_Versione_2_Completa.svg|Visione Completa]]


La parte di inferenza resta uguale più o meno (al netto della gestione del frontend). 
Meglio separare le responsabilità in servizi diversi, ma mettendo i tre servizi sulla stessa macchina... separo le responsabilità in tre servizi ma almeno ho comunque località del servizio.



# Implementazione

Da qui si vede che il GIL non è un problema con la *run* di ONNX: infatti dice esplicitamente che il GIL viene rilasciato per permettere la chiamata di *run* da parte di più thread
https://github.com/microsoft/onnxruntime/issues/11246.
Di base quindi si potrebbero eseguire più modelli senza creare dei processi diversi.
Ad esempio si potrebbe creare un wrap del modello e delle sue componenti che espone un metodo di Run: il metodo di run internamente gestisce un semaforo per limitare il numero di thread che contemporaneamente possono andare a chiamare quel servizio.

https://dagshub.com/Ultralytics/ultralytics/pulls/6583/files?page=0&path=docs%2Fen%2Fguides%2Fyolo-thread-safe-inference.md


Per quanto riguarda il lock questo viene effettivamente rilasciato, ma c'è un aspetto diverso da considerare, cioè il numero di thread che viene prodotto internamente da Onnx per fare l'inferenza. Da documentazione (https://onnxruntime.ai/docs/performance/tune-performance/threading.html#set-number-of-intra-op-threads) ogni sessione alla chiamata di Run crea un numero di thread pari al numero di core fisici... la creazione di questo grande numero di thread crea contesa sulle CPU!

Infatti l'uso del parallelismo non porta beneficio se non si imposta il massimo numero di thread utilizzabili per inferenza.
![[Schermata del 2025-03-21 17-27-05.png|Esecuzione senza set dei thread]]


Se si imposta il massimo numero di thread che può essere creato invece il risultato è quello che segue. 
![[Schermata del 2025-03-21 17-30-09.png|Esecuzione con intra_op = 1]]
In questo caso quindi l'esecuzione sequenziale si mostra davvero più lenta rispetto all'esecuzione parallela proprio perché nel secondo caso abbiamo due thread. Inoltre, a riprova del fatto che il GIL viene rilasciato il tempo di esecuzione con i thread è praticamente uguale al tempo di esecuzione con processi.

Da due thread in poi il vantaggio di esecuzione si assottiglia.
![[Schermata del 2025-03-21 17-40-43.png|Esecuzione con intra_op = 2]]


Per avere l'esecuzione ottimale si dovrebbe analizzare il numero di core e dividerlo per il massimo numero di thread che possono eseguire i modelli, in modo da trovare la combinazione migliore tra numero di thread di inferenza e tracce di esecuzione.

## Sviluppo del FrontEnd
Nell'ottimizzazione del piano ho prodotto due componenti aggiuntive che contengono ognuna un solo nodo, cioè un nodo Generatore e un nodo Ricevitore; queste due componenti sono gestite da un Servizio apposito che diventa proprio il FrontEnd che riceve l'input per la richiesta.
In funzione dell'aggiunta o meno di requisiti alla richiesta (e.g. si vuole scegliere un certo piano per certi requisiti di latenza o di energia), allora si può creare un servizio apposito diverso.

## Bug in ottimizzazione
Parametri del test:
- Due server (server_0 e server_1)
- Bande
	- 0 --> 1 : 125
	- 1 --> 0 : 300
	- 0 --> 0 e 1 --> 1; prese con il monitor (vedere dopo)
- Flops
	- 0 --> $2.5 * 10^9$
	- 1 --> $5 * 10^9$
- Memoria
	- 0 --> 10 MB
	- 1 --> 16 GB

In questa condizione l'ottimizzatore divide il modello in tante piccole componenti.
Per quanto riguarda la modellazione della banda da un server a se stesso ci possono essere due casi:
- I nodi tra cui si trasferiscono i dati fanno parte della stessa componente
	- In questo caso si potrebbe davvero assumere che il tempo di trasmissione sia nullo visto che è gestito dal 
- I nodi tra cui si trasferiscono i dati fanno parte di due componenti diverse
	- In questo caso bisognerebbe considerare il tempo di trasmissione monitorato per andare dal nodo a se stesso (sempre con il meccanismo di ping)

Se entrambe le bande vengono modellate come 0, il problema non sembra presentarsi: bisogna quindi capire se il problema è proprio intrinseco della modellazione. Il problema è che quando si creano così tante componenti l'inferenza si blocca: non so quindi se il blocco è creato dalla gestione dell'inferenza o dalla post elaborazione sulle componenti che in realtà trascura delle dipendenze.

Il problema si presenta quando il valore della funzione obiettivo è superiore a 5 o 5.5.
Nell'immagine si vede il numero di componenti costruite: oltre le 100! Il costo è effettivamente oltre il 5.5 in questi casi.
Ho aggiunto un controllo per testare che il grafo delle componenti non fosse ciclico: di fatto quando non lo è l'implementazione della fase di inferenza non si blocca (vedere dopo).

Facendo alcune prove ho visto che il problema si manifesta nel seguente contesto. Fino ad ora ho fatto gli esperimenti assumendo in fase di ottimizzazione che la banda per passare dati da un server a se stesso fosse infinita (e il conseguente tempo di trasmissione fosse nullo); inserito il monitor ho aggiunto nella risoluzione del problema ANCHE la modellazione del tempo di trasmissione da un sistema a se stesso (vedere considerazione successiva) e questo sembra portare in alcuni casi alla creazione di moltissime componenti. Nel test seguente si vede il contesto di esecuzione di uno di questi casi. 
![[Schermata del 2025-03-17 09-32-34.png|Risoluzione del Problema]]

Potrebbe essere o la prima, la seconda o la terza coppia di stati (più probabile che sia la prima)
![[Schermata del 2025-03-17 09-32-55.png|Stato dei Server]]


La cosa che non mi spiego è il motivo per cui l'ottimizzatore ritenga più conveniente un rimpallo di dati da una parte all'altra piuttosto che la divisione netta: considerando che il server può eseguire più operazioni in Floating Point, non ha molto senso che ci sia questo rimpallo di dati visto che una volta trasferita la computazione comunque l'esecuzione sul server sarà più veloce. In questo senso credo che ci sia qualcosa che manca nella modellazione dell'ottimizzazione.

## Bug in ricerca delle componenti
^bug-ricerca-componenti
Per quanto riguarda il blocco nella fase di inferenza, credo che si verifichi quando c'è ciclicità nel DAG delle componenti: anche questa cosa è strana però visto che la risoluzione delle componenti dovrebbe rimuovere questa ciclicità. Nell'immagine si vede effettivamente un piano prodotto che non è un dag!
![[Schermata del 2025-03-17 10-20-44.png|Piano Prodotto non DAG]]

Caso di Piano Prodotto non DAG: di seguito un caso in cui il grafo prodotto non è un dag e lo stato dei server. A prescindere dal numero di componenti il fatto che il grafo delle componenti non sia un DAG è strano...
![[Schermata del 2025-03-17 10-24-59.png|Ottimizzazione]]

![[Schermata del 2025-03-17 10-28-12.png|Stato dei Server]]

Da qui si può avere una panoramica migliore della situazione e delle interazioni possibili; notiamo che il tempo per andare da server_1 a server_0 è abbastanza più grande rispetto a quello per andare da server_1 a server_1. In questa situazione quindi l'ottimizzatore potrebbe scegliere di fare offloading delle operazioni con il numero di flops più basso verso il device server_0 scegliendo di pagare quel prezzo di trasmissione piuttosto che quello di trasmettere a se stesso quella stessa quantità di dati.

Per quanto riguarda la creazione di un non DAG credo che il problema sia relativo alla gestione dei rami paralleli: probabilmente i controlli sulle dipendenze non sono sufficienti; infatti se tolgo la gestione dei rami paralleli il problema si risolve. Esempio di non DAG: come si vede in entrambe le componenti ci sono dei rami paralleli.
![[Schermata del 2025-03-17 11-09-21.png]]![[Schermata del 2025-03-17 11-10-10.png]]


![[Schermata del 2025-03-17 22-40-51.png]]
In questa figura quello che si vede è che: l'output delle max-pool nella componente (1,32) viene dato come input alla concat della componente (0,30) e fin qui va benissimo. Il problema sorge nel momento in cui aggiungo la mul nella componente (0,30): infatti quando la aggiungo si crea la dipendenza circolare perché il suo output è input proprio della componente (1,32). Il problema probabilmente deriva dall'uso dell'ordinamento topologico: in questo ordinamento le max-pool devono stare prima della concat e la mul deve stare prima sia della concat sia della max-pool; in sintesi quindi abbiamo (mul --> max-pool --> concat). Probabilmente succede questo:
1. Passo per mul e imposto come dipendenza di max-pool e concat la componente di mul
2. Passo per max-pool e imposto come dipendenza della concat quella della max-pool
3. A questo punto la concat ha per dipendenze ((0,30), (1,32)); ma visto che tra le componenti precedenti dirette c'è (0,30), questa componente viene tolta dalle dipendenze e quindi viene inserita in (0,30) creando la dipendenza...



### Bug Fix
^bug-fix-components

Per risolvere il problema ho fatto una modifica dell'algoritmo di ricerca delle componenti.
In particolare si deve fare in modo che un nodo non venga messo in una componente tale che una componente da cui dipende sia a sua volta dipendente. Infatti nel caso della concat il problema è questo: quando scelgo di mettere concat in (0,30) il problema è che non ho nulla a dirmi che c'è già una dipendenza tra (0,30) e (1,32) data dalla dipendenza tra mul e max-pool

Si procede nel seguente modo:
```python
## Componenti da cui un nodo dipende
node_dep_dict : dict[NodeId, set[ComponentId]]
## Componenti possibili che un nodo può prendere (vedere dopo)
node_possible_dict : dict[NodeId, set[ComponentId]]
## Componenti da cui una componente dipende a sua volta (vedere dopo)
comp_dep_dict : dict[ComponentId, set[ComponentId]]

for node_id in graph.topological_order() :
	dependency_set = node_dep_dict[node_id]
	possible_set = node_possible_dict[node_id]

	## Rappresenta l'insieme di componenti che dobbiamo escludere dalle possibili perché creerebbere una dipendenza circolare
	## In particolare queste componenti da escludere sono quelle possibili p tali per cui esiste una componente d da cui il nodo dipende e per cui c'è dipendenza tra p e d
	## Questo significa infatti che inserire il nodo nella componente p creerebbe un ciclo perché (p --> d) di base, ma inserire il nodo x in p farebbe sì che (d --> p) perché x dipende proprio da d
	exclude_set = set()
	for dep_comp_id in dependency_set :
		for poss_comp_id in possible_set :
			if poss_comp_id in comp_dep_dict[dep_comp_id] :
				exclude_set.add(poss_comp_id)

	difference_set = possible_set - exclude_set
	if (difference_set.is_empty() or node_id.is_generator() or node_id.is_receiver()) :
		## Generator and receiver will always be in different components to better handle input and output
		node_comp = generate_new_component()
	else :
		node_comp = difference_set.pick_one_component()

	for desc_id in graph.descendants(node_id) :
		## Tutti i nodi che discendono da questo (i.e. che dipendono direttamente o indirettamente dall'output di questo nodo) hanno la dipendenza da questa componente
		node_dep_dict[desc_id].add(node_comp)

	for next_id in graph.successors(node_id) :
		if node.server_id == next_node.server_id :
			## Se il server è lo stesso, allora il nodo subito successore può essere gestito anche da questa componente
			node_possible_dict[next_id].add(node_comp)

	for parallel_id in graph.parallels(node_id) :
		if node.server_id == parallel_node.server_id :
			node_possible_dict[parallel_id].add(node_comp)
			## Se il server è lo stesso, un nodo parallelo può essere gestito anche da questa componente

	## Espando le dipendenze della componente in cui sto aggiungendo il nodo con quelle del nodo che sto aggiungendo. La rimozione della node_comp permette di assicurare che una componente non sia dipendente da se stessa; infatti per come propagate le dipendenze, la node_comp potrebbe trovarsi anche nel dependency_set del nodo: se così fosse verrebbero sempre scelte delle componenti diverse
	component_dependency_dict[node_comp].expand(dependency_set - node_comp)

```

