La distribuzione del modello quando il modello contiene dei sotto modelli crea dei problemi.

Esempio di modello giocattolo che contiene dei sotto modelli:
- functional_1 è un sotto modello con 3 output
- functional è un modello con 2 input

| Main Model                                      | Functional 1                                    | Functional                                      |
| ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- |
| ![[Schermata del 2024-12-07 22-02-30.png\|200]] | ![[Schermata del 2024-12-07 22-02-44.png\|200]] | ![[Schermata del 2024-12-07 22-02-55.png\|200]] |
I problemi principali di questi sotto modelli sono che:
- Vengono trattati come layer singoli
	- Problemi in termini di calcolo del costo computazionale
- Lo split risulta problematico e solleva errori
- Il livello che rappresenta il sotto modello non ha contezza di esserlo
	- se si fa functional_1.input questo contiene i tensori che risultano dall'input_layer_1 e non quelli che sono ricevuti dall'input_layer e quindi nel modello principale
	- se si analizzano i tensori del sotto modello e si risale ai loro precedenti nella keras_history, questa history si ferma all'input layer del sotto modello e non tiene in conto le operazioni precedenti del modello genitore

Per gestire queste complessità, si è pensato di sostituire i livelli di input dei sotto modelli con degli *IdentityLayer* e i livelli di output dei sotto modelli mandano il loro risultato ad un *IdentityLayer*: questi livelli sono dei placeholder che servono a semplificare la ricostruzione del modello unnested; da un punto di vista di costo non dovrebbero avere costo eccessivo (ammesso ne abbiano) visto che da documentazione prendono il loro input e lo restituiscono in output

Il problema principale che c'è per ricostruire il modello unnested è il fatto che per poter ottenere un modello non annidato la ricostruzione delle operazioni tramite i tensori di input/output dei livelli del modello non va bene. Supponiamo di avere uno dei due livelli dense di functional_1: ammesso di ricostruire che il suo input viene da *input_layer* quando si chiama `dense(kerasTensor)` (e questo si deve chiamare per poter ottenere un tensore che abbia una history diversa), otteniamo un tensore diverso da quello che è dato da `dense.output`; essendo diverso, per poter continuare dobbiamo eseguire tutti i livelli successivi che derivano de questo nuovo tensore, per ottenere dei tensori che siano compatibili con la nuova history.
In conclusione bisogna rieseguire tutto il modello per creare la nuova successione di tensori.

Operazioni che si fanno:
1. Trovare le operazioni precedenti per ogni operazione
2. Trovare tutte le operazioni
3. Trovare gli input layer del modello principale
4. Trovare gli output del modello principale
5. Trovare tutti i sotto modelli
6. Ricostruire l'esecuzione del modello facendo l'unnest dei sotto modelli


## Ricerca operazioni precedenti
Viene fatta cercando prima le operazioni successive di ogni operazione e poi invertendo le connessioni trovate.

### Ricerca operazioni successive
In questa implementazione abbiamo che:
- Se un livello è un livello di output di sotto modello, questo viene connesso ad un livello che ha lo stesso nome del sotto modello
	- Si tratterà del livello identità che aggregherà gli output del sotto modello e che permetterà di mantenere la struttura più o meno uguale in termini di connessioni
- Se un'operazione ha per prossima operazione un sotto modello
	- Devo collegare l'operazione all'input_layer corrispondente nel sotto modello
```python
nextOpsDict <- maps opName on nextOps names
opQueue <- model.operations (all unnested operations of model)

while opQueue :
	currOp <- extract next op
	if currOp is Model :
		subNextOpsDict <- find sub connections recursively
		nextOpsDict.extend(subNextOpsDict)
		for op in subNextOpsDict:
			if op is output op:
				add currOp.name as next of op
	for nextOp in currOp._outbound_nodes :
		if nextOp not model:
			add next normally
		else :
			# Have to find the corresponding input layers
			subModelInputLayers <- find corresponding
			Add next ops of currOp as found sub model input layer
```

La ricerca dell'input layer del sotto modello che corrisponde ad un livello nel sovra modello è fatta combinando due cose:
- `subModel._inbound_nodes[0].arguments._flat_arguments`
	- Contiene tutti gli argomenti che il sub model riceve nel ruolo di layer (e quindi dai suoi predecessori nel sovra modello)
	- Posso prendere i livelli che li hanno generati tramite la _keras_history
- `subModel.inputs`
	- Contiene gli argomenti che il sub model riceve come modello (e quindi dai suoi input layer)
	- Posso prendere gli input_layer corrispondenti tramite keras_history

Considerando che se il modello ha più di un input, allora ha più input layer, allora quello che faccio è che IN ORDINE mi scorro gli elementi di flat_arguments e quando trovo un livello che corrisponde al livello predecessore che sto considerando, allora gli assegno il livello di input corrispondente allo stesso indice dell'elemento nel flat_arguments

> [!Warning] Ordine
> L'uso dell'ordine potrebbe creare dei problemi in caso ci fossero delle operazioni strane e/o dei modi non convenzionali di passare gli input


## Ricerca Operazioni, Sotto Modelli, Input Layers
Fatti sempre tramite delle funzioni ricorsive che analizzano le operazioni del modello e ne vedono il tipo

## Ricostruzione del Modello
Per ricostruire il modello si devono eseguire tutte le operazioni al fine di ricostruire dei tensori con la history coerente:
- Gli output prodotti sono inizializzati a quelli degli input_layer principali, cioè quelli del sovra modello
- Si eseguono le operazioni una per una se non già eseguite
- Si ricostruisce un modello che ha:
	- Input --> Stesso input del Sovra Modello
	- Output --> Stesso output del Sovra Modello
		- Preso dai ProducedOutputs!! In questo modo la sua keras_history è compatibile con quella che mi serve per deannidare il modello
```python
producedOutputs <- maps operation names on its produced outputs
for inpLayer in main model input layer list :
	## Init process
	producedOutputs[inpLayer.name] = inpLayer.output

for op in allOps :
	if op.name not in producedOutputs :
		runOperation(op)
```

### Esecuzione operazione
Ricorsivamente devo assicurarmi che tutte le operazioni precedenti siano state eseguite.
Sono poi fatte tre operazioni:
- Wrap dell'operazione
- Ricerca della struttura degli argomenti della chiamata
- Preparazione dell'input per la chiamata
```python
## Check if all dependencies have been satisfied
for prevOpName in prevOps[currOp] :
	if prevOpName not in producedOutputs:
		runOperation(prevOp)

toCall <- wrapOperation(currOp)
callArgs <- findArguments(currOp, allSubModels)
opInput <- unpackArguments(callArgs)

opOutput = toCall(opInput)
producedOutputs[currOp.name] = list(opOutput)
```

#### Operation Wrap
Se l'operazione è un sotto modello oppure un input_layer allora questa operazione viene sostituita con un *IdentityLayer*.
Questi Identity layer servono a mantenere inalterata la struttura del modello e a non creare problemi e complicazioni nella ricerca dell'operazione precedente di una.

Se l'operazione non rientra tra questi tipi viene ritornata lei stessa.

Notare che se un livello è un input_layer potrà essere l'input layer solo di un sotto modello visto che stiamo nella chiamata run e che vengono eseguite solo operazioni che non sono in producedOutputs; visto che i livelli di input principali li abbiamo aggiunti all'inizio delle chiamate non posso avere i livelli di input principali qui dentro.

#### Find Arguments
In questa fase lo scopo è di trovare la struttura degli argomenti della chiamata: questa struttura sarà poi riempita dai tensori prodotti dalle operazioni precedenti.

Tre casi:
- L'operazione è un Model (quindi un sotto modello)
	- In questo caso stiamo sostituendo l'operazione con un IdentityLayer che restituisce in output gli output del sotto modello
		- Quindi gli args hanno la struttura di `currOp.outputs`
- L'operazione è normale
	- Gli args hanno la stessa struttura di `operation._inbound_nodes[0].arguments.args`
- L'operazione è un input layer
	- Sto sostituendo con un IdentityLayer che restituisce in output l'output di uno dei livelli precedenti
		1. Trovo a quale sotto modello appartiene questo input_layer
		2. Trovo l'indice dell'input layer nell'elenco di input layer (serve se il sotto modello ha più di un argomento in input)
		3. Prendo ogni elemento in `opSubModel._inbound_nodes[0].arguments.args:`
			1. Se è una lista, allora il sotto modello ha più input e prendo quello corrispondente all'indice che ho trovato al passo prima
			2. Se è singolo allora ritorno l'unico argomento


> [!Warning] Ricostruzione sull'input Layer
> Potrebbe non essere abbastanza generalizzante


#### Unpack Arguments
QUESTO PUNTO RAPPRESENTA IL RACCORDO TRA LA HISTORY PRECEDENTE E QUELLA CHE SI STA COSTRUENDO ADESSO
 
Per ogni argomento nell'args che ho ricostruito:
- Se è un KerasTensor (questi sono ancora quelli del modello originale!!)
	- Prendo la sua history
	- Dalla history trovo: operazionePrecedente e tensor_index
		- Il tensor_index rappresenta l'indice del tensore nell'output dell'operazione precedente da cui questo tensore deriva
	- Prendo da processedOutputs l'output dell'operazione precedente e il tensore corrispondente a tensor_index
- Se è una lista (potrebbe essere per operazioni che hanno più di un input)
	- Chiamo ricorsivamente il metodo di unpack
- Se è None
	- Continua
- Se è altro (ad esempio una costante)
	- Append normale

## Risultato
### Modello Giocattolo
![[Schermata del 2024-12-07 23-20-38.png|Modello Unnested]]


Come si vede dalla foto i livelli identità sostituiscono gli input_layer e i sub model layers che erano invece nei modelli precedenti.
L'identity_layer che corrisponde a functional_1 ad esempio riceve proprio l'input dagli stessi livelli che prima erano output del sotto modello.


### Yolo V8 (Keras_Cv)
Si vede che:
- L'input layer della backbone è sostituita dal livello Identità
- L'output della backbone viene raccolto nel livello Identity da cui poi parte verso gli altri livelli che lo vogliono in input

| Identity InputLayer BackBone                    | Identity Output BackBone                        |
| ----------------------------------------------- | ----------------------------------------------- |
| ![[Schermata del 2024-12-07 23-27-39.png\|300]] | ![[Schermata del 2024-12-07 23-29-03.png\|300]] |


## Bug - Sequential Nested
Le istanze di *Sequential*, anche se l'input_layer viene inserito in modo esplicito non hanno InputLayer all'inizio, quindi il parsing non si può fare sulla base della presenza degli input_layer.
![[Schermata del 2024-12-17 15-12-01.png | Sequential Model]]

Inoltre le istanze di Sequential non hanno gli attributi:
- *operations*
- *output_names*
![[Schermata del 2024-12-17 15-18-32.png|Assenza *operations*]]![[Schermata del 2024-12-17 15-19-11.png|Assenza *output_names*]]
Questi tre aspetti portano l'Unnest a non funzionare se abbiamo delle classi Sequential innestate.

### Bug Fix
Per risolvere questo si astraggono questi attributi in funzioni che distinguono a seconda del tipo:
![[Schermata del 2024-12-17 16-23-25.png|Funzioni per Trovare le Info Necessarie]]

Fare questo permette di risolvere il problema ed estendere anche a modelli sequenziali.


## Bug - Input Order
Ci sono dei casi in cui ci sono delle discrepanze tra l'ordine con cui il livello (o il sotto modello) manda fuori il suo output e i tensor_index della KerasHistory.

La KerasHistory è formata da tre parti:
- Operation
- Node Index
- Tensor Index

In particolare sembra che:
- Node Index corrisponda all'indice dalla lista di inbound_nodes `operation._inbound_nodes[node_idx]`
- Dato un inbound_node, questo presenta tra i suoi attributi:
	- arguments
		- Usato per ricostruire gli argomeni
	- outputs / output_tensors
		- Che contiene i tensori dati in output dall'operazione in question
		- Nello specifico il tensor_idx sembra riferirsi proprio all'output come rappresentato in questa lista, quindi corrisponde a `operation._inbound_nodes[node_idx].outputs[tensor_idx]`
		- Questa lista di output sembra essere ordinata per `kerasTensor.name`

In figura abbiamo due ordini diversi tra:
- InboundOut: qui si vede come i tensori dell'output sono ordinati per name
- CallOut
![[Schermata del 2024-12-18 21-12-47 1.png|Esempio del Bug]]

![[Pasted image 20241218212058.png|Differenza tra Input voluto e ricostruito]]

### Bug Fix
Per risolvere il bug, visto l'apparente ordinamento della lista di outputs, una volta che viene fatta la `opOutput = toCall(*args, **kwargs)`, la `opOutput` viene convertita in una lista ordinata di KerasTensor, dove l'ordinamento è fatto proprio per `kerasTensor.name`; questo viene inserito all'interno della funzione `convertToList`.
![[Schermata del 2024-12-18 17-34-44.png|*convertToList* Modificata]]

> [!NOTE] 
> Questa aggiunta sembra risolvere il problema indicato, ma ci sono dei modelli di Segmentation per cui non è sufficiente, come ad esempio SAM. In questo caso bisogna ancora capire un attimo cosa sta succedendo e il motivo per cui il modello non funziona come dovrebbe.

## Bug - Riuso di Layers
Ci sono dei modelli (vedere [[Segmentation]]) in cui alcuni InputLayer vengono riutilizzati come input layer dei sotto modelli: questo porta alla rottura dell'unnest che si basa sull'unicità dei nomi dei livelli.
![[Schermata del 2024-12-18 22-30-28.png|Successori Ricorsivi]]

### Bug Fix
Per risolvere questo problema si possono costruire dei Wrapper per le operazioni; a questi Wrapper vengono poi associati dei nomi unici in formato di path (ad esempio "/mainMod/subMod_1/subMod_2/layerName").
In questo modo, anche se il layer è stato riutilizzato, il suo nome sarà unico.
A questo punto quando creo l'IdentityLayer associato a questo InputLayer lo creerò con il nome nuovo ma con la stessa funzionalità: questo IdentityLayer quindi prenderà in input l'output dell'InputLayer originario. Questo in teoria dovrebbe bastare a risolvere il conflitto.

Questo problema in realtà si può ripetere anche per altri tipi di livello: Keras permette il riutilizzo di un modello all'interno dello stesso modello e in suoi sotto modelli.
All'interno di `operation._inbound_nodes` sono presenti le istanze di nodes dell'operazione: significa che se all'interno del modello quel livello viene usato tre volte, all'interno della lista ritornata ci saranno tre istanze della classe Node, ognuna corrispondente ad un nodo con cui quel livello compare nel grafo di computazione del modello.

Quando accediamo alla KerasHistory di un tensor, questa History contiene:
- Operation
- NodeIndex : rappresenta l'indice del nodo all'interno della lista di _inbound_nodes
- TensorIndex: rappresenta l'indice dell'output dato in risultato dall'operazione in questione
Questo perché anche se l'operazione è la stessa i KerasTensor che sono prodotti sono diversi perché prodotti in due fasi diverse e con input diversi! In sostanza abbiamo una matrice indicizzata da (nodeIdx, tensorIdx) e che mi permette di ottenere l'output dato da quel nodo.

Per gestire questi casi più complessi quindi, non è sufficiente accedere all'indice 0 della lista di _inboun_nodes_, ma è necessario accedere indicizzando per nodeIdx.

Un'altro aspetto da considerare è quello delle connessioni: nell'andare avanti (sfruttando quindi gli outbound_nodes) non è scontato che si possa capire a quale istanza di Node si fa riferimento per quella operazione; andando indietro invece (sfruttando gli inbound_nodes) si può farlo usando appunto le keras_history.

A questo punto credo che tutto debba essere fatto ragionando per Node: in questo senso quindi, quando mi salvo un'operazione prendo anche il node_idx dato dalla keras history.


## Post Scoperta Livelli Ripetuti
Per rendere l'implementazione più tollerante alla ripetizione dei livelli ho costruito delle astrazioni che permettessero di gestire questo aspetto. Il wrap non è più fatto di Operazioni (che appunto possono essere ripetute all'interno del modello), ma di Nodes, che invece non vengono ripetuti.


### NodeKey
Rappresenta una chiave univoca associata al nodo. Questa classe quindi si occupa di fare il Wrap di una tupla che definisce il nodo in maniera univoca (o almeno dovrebbe): è stata creata un classe per avere più flessibilità al cambio della definizione della chiave.

> [!Warning]
> Per adesso è stata definita solo come una coppia (operationName, nodeIdx). Questa definizione in teoria non è sufficiente per gestire casi complessi di annidamento e ripetizione dell'operazione. 
> 
> In teoria dovrebbe essere costruita come una tupla che definisce una sorta di path all'interno del modello, del tipo (mainMod, subMod_1.name, subMod_1.idx, subMod_2.name, subMod_2.idx, op.name, op.idx). Una definizione di questo tipo ci permetterebbe di poter identificare un nodo in modo univoco all'interno del grafo.

Supponiamo che vi sia un sotto modello *subMod* usato n volte e che l'operazione *op* sia all'interno di questo modello e sia usata una sola volta al suo interno. In questo caso, sebbene l'operazione sia di fatto usata più volte, perché eseguita n volte, c'è un unico inbound_node ad esso associato. Per poter distinguere quindi tra i path per arrivare al nodo della *op* nelle varie ripetizioni di *subMod*, è necessario tenere traccia dell'istanza di *subMod* a cui il nodo che stiamo considerando appartiene.
In questo senso quindi una chiave definita come detto ci permette di definire un vero e proprio path e di creare dei NodeWrapper diversi.


### NodeWrapper
Fa il wrap di un Nodo del modello, permettendo di accedere ai vari attributi dei nodi, nascosti e non da un unico punto; in questo modo, qualora dovesse cambiare il modo in cui Keras definisce il suo grafo, si dovrebbe riuscire a gestire il cambiamento in modo più centralizzato.

Tra le varie informazioni mantenute dalla classe abbiamo:
- Node di cui fa il Wrap
- NodeIdx, ovvero l'indice dell'inbound node
- Model, ovvero il modello a cui il nodo corrente appartiene
	- Per comodità
- modelKey, per rintracciare il nodo del modello in modo semplice e per costruire le chiavi in modo iterativo (qualora servisse)
- nodeKey, identificatore univoco del node all'interno del grafo del modello
### NodePool
Rappresenta una Pool di NodeWrapper.
Si occupa di fare il wrap di un dizionario che mappa la NodeKey --> NodeWrapper.

Affinché la NodeKey si potesse usare come chiave le è stato fatto implementare le funzioni di *hash* e *eq*. 

Il metodo fondamentale del pool è il `addNodesFromOperation`: questo metodo permette di aggiungere tutti i nodi associati ad un'operazione e che fanno parte del sotto modello corrente. I parametri sono:
- operation, l'operazione i cui nodi vogliamo aggiungere
- model, il modello di cui l'operazione fa parte
- modelKey, la chiave associata all'operazione del modello (per poter costruire le chiavi dei nodi dell'operazione corrente).
Per prima cosa vengono presi tutti i nodi che appartengono al modello di appartenenza dell'operazione e, per ogni nodo in `_inbound_nodes` dell'operazione si vede se ricade in questi nodi: se ricade allora viene aggiunto al ModelPool qualora non sia ancora stato inserito.

Gli altri due metodi significativi sono i `findInputNodesKeys` e `findOutputNodesKeys` che permettono di trovare gli input nodes e gli output nodes del main model; siamo interessati solo al main model perché di fatto stiamo costruendo il modello Unnested.

### ModelGraph
Costruisce il grafo del modello dato in input.

L'inizializzazione della `NodePool` viene fatta in modo ricorsivo: per ogni operazione del modello si fa l'aggiunta alla nodePool; se il nodo aggiunto è associato ad un modello si fa la ricorsione su quello. Contestualmente viene anche inizializzata una lista di `depthSortedKeys` che ci ridà la chiavi aggiunte alla nodePool in ordine di tempo. 
Per quanto riguarda la modelKey passata al metodo:
- Quando chiamata all'inizio si passa None, assumendo che al modello base non sia necessario dare chiave
- Quando chiamata sul sotto modello, si passa la chiave del sotto modello stesso, che conosciamo perché ritornata dall'aggiunta dei nodi nel nodePool.

#### FindPrevConns
Si tratta di uno dei metodi fondamentali: permette di ricostruire le connessioni del grafo. Anche in questo caso non lavoriamo per operazioni, ma per nodi nel grafo di computazione.

Abbiamo tre casi, a seconda della natura del currentNode:
- KerasModel.  In questo caso l'input viene ricevuto dai livelli di output del sotto modello stesso
	1. Si prende l'output del sotto modello, ovvero i suoi tensori di output
	2. Per ognuno di questi tensori, tramite keras_history, si vede qual è il nodo che li ha generati
	3. Si costruisce la chiave di questi nodi
		1. Notare che questa chiave sarà data da currentNodeKey + outNode.op.name + outNodeIdx. In particolare dobbiamo prendere come punto di partenza il currentNodeKey visto che il nodo i questione è in questo sotto modello
	4. Prendiamo tutti i nodi associati alle chiavi che abbiamo costruito
- InputLayer. In questo caso l'input viene ricevuto dai livelli che mandano l'input al nodo del sotto modello.
	1. Si prende l'input del nodo associato al sotto modello ( i suoi tensori di input)
	2. Tramite keras_history si ricostruiscono i nodi che hanno prodotto questi nodi
		1. Questi nodi appartengono al modello che ha il sotto modello come appunto sotto livello, quindi la chiave che definisce questi nodi sarà generata con la  `ownerNode.getOwnerKey()` come punto di partenza.
		   Supponiamo ad esempio di avere l'input layer del sotto modello con chiave `(subMod_1, 0, subMod_2, 0, subMod_3, 0, input_layer_1, 0)`; l'input che viene dato ad input_layer_1 proviene da un nodo che si trova in subMod_2, quindi la chiave di partenza deve essere `(subMod_1, 0, subMod_2, 0)`, ovvero proprio la chiave dell'owner dell'owner dell'input_layer_1.
	3. Prendiamo i livelli di input del sotto modello
	4. Cerchiamo la corrispondenza tramite indice (QUESTA COSA POTREBBE NON FUNZIONARE IN ALCUNI CASI)
- OtherLayer
	1. Prendiamo i tensori input dell'operazione corrente
	2. Prendiamo i nodi generatori tramite keras_history
	3. Costruiamo le chiavi
	4. Prendiamo i Wrapper corrispondenti tramite nodePool

### Reconstructor
Funziona più o meno nello stesso modo di prima, solo che lavora per Nodi e non per operazioni.

Procedimento:
1. Inizializziamo gli output prodotti. Inseriamo dei tensori per gli input layer del modello
2. Per ogni nodo
	1. Eseguiamo i suoi nodi predecessori
	2. Facciamo eventuale wrap dell'operazion
		1. Se si tratta di un InputLayer o di un sub model lo sostituiamo con un identity layer avente per output l'input dell'operazione di cui facciamo il Wrap
	3. Cerchiamo la chiave per i nodi che generano l'input per il nodo corrente
		1. Se è un nodo normale, sarà un nodo nello stesso ownerModel, quindi è la chiave dell'ownerModel 
		2. Se è un nodo di input sarà un nodo nell'owner dell'owner
		3. Se è un nodi di Model sarà un nodo nel modello stesso, quindi è la chiave del nodo che stiamo elaborando
	4. Facciamo l'unpack sia di *args* che di *kwargs*
	5. Eseguiamo l'operazione
	6. Salviamo l'output dell'operazione come una lista
3. Costruiamo l'input e l'output del modello finale 
	1. Costruiti tramite due dict, inputOpsDict e outputOpsDict. Ogni dizionario mappa, per un certo nodo/operazione, qual è l'indice del tensore di output che si vuole come input (per evitare di mandare in input/output tensori superflui).

### Risultato
Fatta in questo modo la ricostruzione riesce a gestire anche il riuso di layer all'interno del modello. Come si vede in figura il modello in questione presenta due volte il sotto modello sequential e un livello *dense* ripetuto una volta nel modello principale e una volta nel modello *functional_1*. L'implementazione sì fatta riesce a fare l'unnest del modello e ad ottenere due modelli equivalenti in termini di risultato.
![[Schermata del 2024-12-22 19-09-28.png|Unnest su un modello con ripetizioni|400]]


> [!Warning] 
> Fare la ricostruzione del modello in questa maniera implica che vengono aggiunti dei nuovi nodi nel grafo originale... Infatti la \_\_call\_\_ che facciamo viene mappata su una *symbolic_call* internamente e questa call aggiunge dei nodi all'interno del grafo.
> Comunque questo non sembra creare problemi: quando viene ricaricato il modello mantiene solo i suoi nodi e non altri.


