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