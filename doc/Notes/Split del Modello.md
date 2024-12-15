## Divisione di Modello Generico (Focus su MobileNetV3)

Riferimento al link https://gist.github.com/martinsbruveris/1ce43d4fe36f40e29e1f69fd036f1626.

```python
model = build_model()

cut_layer = model.get_layer(name="fc2")
model_a = Model(model.inputs, cut_layer.output)
model_b = Model(cut_layer.output, model.outputs)
```

Prendendo il layer tramite il suo nome e passando gli input e gli output del layer nelle divisioni del modello, si ottengono dei sotto modelli del modello originale che fanno la computazione solo di quel pezzo.
Considerazioni :

- Gestione dei nodi di fork e join
  - Se abbiamo delle pipeline parallele all'interno del modello, bisogna fare attenzione a dove vengono messi i nodi di fork e di join: se il subModel comincia con il nodo di fork e finisce con il nodo di join, tutto ciò che è intermedio viene inserito nel sotto modello.
  - Si potrebbe in realtà riuscire a fare una divisione abbastanza tranquilla in ogni punto della rete: supposto che i nodi di fork e join siano in server diversi, quando il fork esegue invia il suo input al join e alla parte di esecuzione dell'altro; quando il join riceve, deve aspettare prima la seconda parte di esecuzione, quindi ritorna null al nodo di fork; quando il join riceve la seconda parte di input allora emette il risultato e terminata l'esecuzione totale, il fork riceverà l'output dal secondo ramo.
    - NOOOO. Questa cosa non funziona (vedere [[Split_On_Join.png]]): per arrivare all'output di _multiply_ c'è bisogno dell'output di entrambi i rami, quindi se parto da _Conv2D_, keras mette in automatico anche tutto l'altro ramo che serve per arrivare all'iput per quello specifico livello
    - Questa cosa funziona se si specificano come input del modello quelli provenienti da tutti i nodi di input e che non sono nel sotto modello (vedere [[Multi_Input_Variant.png]] e [[Multi_Input_Variant_Graph.png]]), ad esempio specifico come input sia l'output della relu (che diventa input della multiply) sia l'output dell'AvgPooling (che diventa inpur di Conv2D).
    - Allo stesso modo funziona se il livello di partenza è un livello di Join (vedere [[Starting_With_Join.png]]) quindi ad esempio parto dalla Multiply e vado alla BatchNormalization.
- La gestione dei nodi che non sono layer (come per operazioni numpy) non dovrebbe più dare problemi: se si fa la divisione del modello usando SOLO i layers restituiti da _model.layers_ quei nodi del grafo NON sono annoverato tra i layer quindi:
  - lo split lì non si può fare.
  - Allo stesso tempo saranno considerati come parte della computazione per arrivare al nodo di output, quindi verranno presi in considerazione in automatico.

| Fork & Join             | Split on Join senza input   |
| ----------------------- | --------------------------- |
| ![[Fork-Join.png\|300]] | ![[Split_On_Join.png\|300]] |

```python
subModel = keras.Model(
	inputs=[
		model.get_layer("expanded_conv_11_squeeze_excite_conv").input,
		model.get_layer("activation_12").output],
	outputs=model.get_layer("expanded_conv_11_squeeze_excite_mul").output,
)
```

| Inizio con Nodo di Join     | Split on Join con Input aggiuntivi                     |
| --------------------------- | ------------------------------------------------------ |
| ![[Starting_With_Join.png]] | ![Variante Multi Input](Multi_Input_Variant_Graph.png) |

Posto di fare molta attenzione a dove e come il modello viene tagliato e quali e quanti sono gli input che si aspetta di ricevere.
In linea di principio:

- Se il sotto modello è sequenziale non ho problemi
- Se il sotto modello contiene il fork ma non il join non ho problemi: devo inviare l'output del sotto modello a tutti i successori (che comunque devo di trovare)
- Se il sotto modello contiene il join, ma non il fork devo fare attenzione a mettere nell'input del sotto modello gli output di tutti i rami che non sono compresi nel sotto modello (che comunque devo trovare)
- Se il sotto modello contiene sia il fork sia il join corrispondente non ho problemi

Devo comunque fare attenzione a ricostruire da quali livelli mi aspetto di ricevere l'input per poter recuperare le informazioni sui loro tensori, in particolare se il sotto modello contiene dei join.

Pseudocodice

```python
for layer in subModel:
	layerInputs <- get input layers
	for inputLayer in layerInputs:
		if inputLayer in subModel:
			continue
		else:
			add inputLayer.output as input of the submodel
```

Dato un nodo devo trovare i primi layer validi che lo precedono e che lo succedono.

```python
### Cerco i nodi predecessori e successori
for layer in model:
	inputOps <- get layer input ops
	prevLayers = []
	for op in inputOp:
		if op is a layer:
			prevLayers.append(op)
		else :
			opPrevLayers <- get op prev layers
			prevLayers.append(opPrevLayers)
		## Updating prev layers
		layer.prevLayers = prevLayers
		## Updating next layers
		for otherLayer in prevLayers :
			otherLayer.nextLayers.append(layer)

```

La divisione del modello fatta in questo modo funziona!!!

Bisogna un attimo ricostruire però il rapporto tra nomi dei tensori di input e nomi dei tensori di output. Il codice di seguito crea un raccordo tra nome del livello precedente e nomi dei tensori in input al sotto modello da quegli stessi livelli.

```python
subModelInput += [
	keras.Input(
		shape=prevLayerOut[i].shape[1:],
		tensor=prevLayerOut[i],
		name=prevLayer,
		)
	for i in range(0, len(prevLayerOut))
	]
```

| Output Sotto-Modello Precedente               | Input Sotto-Modello Successivo    |
| --------------------------------------------- | --------------------------------- |
| ![[Output Sotto Modello Precedente.png\|280]] | ![[Input Sotto Modello.png\|307]] |


Funziona ma parzialmente: bisogna fare in modo che gli input e gli output dei modelli siano dei tensori con nomi, in modo da poter raccordare l'output di un modello con l'input del modello successivo.
RISOLTO: Creati dei dict per rappresentare input ed output e assegnati i nomi corrispondenti che servono.

> [!Bug] Non era risolto in realtà...
>
> > [!Done] Risolto
> > Risolto creando prima il nuovo oggetto di tipo keras.Input partendo dal tensore output del livello precedente e poi modificandone il nome con _obj.name_.
> >
> > > [!Warning] Warning ricevuto nel Parsing
> > > Durante il parsing del modello e la ricostruzione dei dizionari di input e di output viene ricevuto un Warning. In sostanza c'è una mancata corrispondenza tra i nomi dei tensori di input nel dizionario e le chiavi del dizionario stesso; comunque Keras sembra gestire la cosa abbastanza bene, ricostruendo il tutto correttamente.
> > > La cosa strana è che se cerco di risolvere il problema impostando un nuovo tensore di input ho errore di operazione ripetuta.
> > >
> > > Non so se potrebbe dare problemi in altri punti dell'esecuzione o nel parsing di modelli diversi.
> > > Comunque sì: c'era un problema nel nome dell'input quando il sotto modello era esportato come saved_model.
> >
> > Con questa soluzione sembra funzionare sia per formato .keras sia per formato saved_model
>
> Quando si crea una nuova lista con i diversi modelli veniva dato un errore al salvataggio del modello. Probabilmente era dovuto al fatto che quel tensore veniva trasformato in un placeholder.

```
### Warining
/opt/conda/lib/python3.11/site-packages/keras/src/models/functional.py:106: UserWarning: When providing `inputs` as a dict, all keys in the dict must match the names of the corresponding tensors. Received key 'activation_18' mapping to value <KerasTensor shape=(None, None, None, 960), dtype=float32, sparse=False, name=keras_tensor_188> which has name 'keras_tensor_188'. Change the tensor name to 'activation_18' (via `Input(..., name='activation_18')`)

#############

ValueError: The name "expanded_conv_3_expand" is used 2 times in the model. All operation names should be unique.

```

Il secondo errore qui era dovuto al fatto che in realtà il modello non era diviso correttamente, ma ripartiva da capo tutte le volte, portando quindi a ripetizione delle operazioni.

> [!Bug] In realtà non risolveva il problema
> Codice che risolve il problema
>
> ```python
> if len(prevOpsDict[layer.name]) == 0:
> 	## Input Layer
> 	newInput = keras.Input(
> 		shape=layer.output.shape[1:],
> 		tensor=layer.output,
> 		name=layer.name,
> 	)
> 	newInput.name = layer.name
> 	subModelInput[layer.name] = newInput
> else:
> 	for prevLayerName in prevOpsDict[layer.name]:
> 		prevLayerOut = model.get_layer(prevLayerName).output
> 		if prevLayerName not in subLayersNames:
> 			newInput = keras.Input(
> 				shape=prevLayerOut.shape[1:],
> 				tensor=prevLayerOut,
> 				name=prevLayerName,
> 			)
> 			newInput.name = prevLayerName
> 		subModelInput[prevLayerName] = newInput
> ```

Per adesso risolto così, ma il warning rimane, anche se non dovrebbe dare problemi.

```python
subModelInput = {}
for layer in subLayers:
	if len(prevOpsDict[layer.name]) == 0:
		subModelInput["input"] = layer.output
	else:
		for prevLayerName in prevOpsDict[layer.name]:
			prevLayer = model.get_layer(prevLayerName)
			prevLayerOut = (
				prevLayer.output
				if isinstance(prevLayer.output, list)
				else [prevLayer.output]
			)
			if prevLayerName not in subLayersNames:
				for _, out in enumerate(prevLayerOut):
				# newInput = keras.Input(tensor=prevLayerOut)
					subModelInput[out.name] = out
return subModelInput
```

> [!NOTE] Distribuzione per Livello Singolo
> Notare che questa stessa distribuzione si può fare nello stesso modo se si vuole dividere il modello per livelli singoli: è sufficiente mettere 1 come size massima della sotto lista di layers.

> [!Done] Nuova Divisione del Modello
> Piuttosto che fare la divisione partendo dal _config_ del modello (in caso di modelli strani / complessi il config viene esportato in modo strano quindi non funziona), uso direttamente gli attributi degli oggetti keras!!
> Posso trovare i livelli validi predecessori del modello direttamente partendo dai kerasTensor e dalle loro History
>
> > ```python
> > kerasTensor._keras_history ## Gets its operation history
> > kerasTensor._keras_history.operation ## Gets list of previous operations
> > model.output._keras_history.operation.input ## Gets operation input (it is a keras tensor!!)
> > ```
>
> Funziona decisamente meglio e non dovrebbe dare problemi di parsing del config qualora ci fossero modelli con config non standard

> [!bug]
> Per modelli più complessi l'output di un layer potrebbe essere una lista o un dizionario: in questo caso bisogna gestire vedendo quale output di questo livello viene ricevuto come input del livello corrente. Potrebbero esserci altri problemi dovuti ai nomi di questi input però...
> Vedere esempio fatto con KerasCV e YoloV8.
>
> C'è differenza tra un layer che dà il suo output a più di un livello successivo e un livello che ha proprio più input. In questo caso il Parsing fatto non funziona perché assume un unico output

| Errore in caso di livello con più Keras Tensors di output |
| --------------------------------------------------------- |
| ![[Problema Livello Con Multi Output.png]]                |

Modificato il parsing in modo che supporti delle liste, ma per adesso se un layer ha più di un output non posso recuperare quale output di quel layer sia usato come input del layer corrente.
All'interno dell'oggetto KerasHistory c'è l'attributo *tensor_index* che potrebbe rappresentare l'indice del tensore di output dell'operazione precedente... (per adesso è solo una supposizione).

| KerasHistory e TensorIndex                 |
| ------------------------------------------ |
| ![[Schermata del 2024-11-29 13-42-00.png]] |
Conferma di questo!!
Il ripercorrendo la storia del livello tre fino a *concatenate_5* troviamo functional con indici 0 e 1; se vediamo la keras_history il tensor_index è proprio lo stesso!! Quindi se il sotto modello ha più di un output possiamo trovare a quale tensore corrisponde prendendo il tensor_index nella history. Posso convertire il dict in una list ed indicizzare tramite il tensor index

| Summary       | ![[Schermata del 2024-12-02 16-06-28.png]] |
| ------------- | ------------------------------------------ |
| **History**   | ![[Schermata del 2024-12-02 16-06-57.png]] |
| **Histories** | ![[Schermata del 2024-12-02 16-56-53.png]] |

Da provare se l'accesso in figura successiva funziona

| Accesso all'output ad indice               |
| ------------------------------------------ |
| ![[Schermata del 2024-12-02 17-18-28.png]] |


### Gestione di Sotto Modelli
In alcuni casi un layer potrebbe non essere un layer in senso stretto, ma potrebbe essere un sotto modello inserito all'interno di un altro modello. In questo caso la ricostruzione dei livelli precedenti tramite keras_history si rompe.

Per provare a risolvere la cosa si cercano prima tutti i livelli in modo ricorsivo e costruendo una lista con tutti i livelli effettivi del modello. Tutte le operazioni che prima si facevano sul model.layers adesso si devono spostare su questa lista di livelli.

Se il modello non ha dei sotto modelli il problema non si pone perché la lista ottenuta sarà uguale a quella che si otterrebbe con model.layers.


> [!Warning] Dipendenze
> Bisogna fare attenzione ad eventuali dipendenze cicliche che si potrebbero introdurre dividendo il modello

Se un livello riceve l'output di un sotto modello, l'input ricevuto non ha come history il livello di uscita del sotto modello, ma ha come history il livello che rappresenta il sotto modello: in figura, il primo è il livello predecessore, mentre il secondo è l'output del sotto modello.

| Differenza tra input e output del sotto modello |
| ----------------------------------------------- |
| ![[Schermata del 2024-12-02 13-29-13.png]]      |
Possibile Gestione:
- Si potrebbe dire che se un livello è un sotto modello, allora si prende il suo output come input
- Si considera come predecessore più prossimo l'output del sotto modello e si procede con la ricerca della history
- Si considera il sotto modello separatamente dal modello principale: si dividono i sotto modelli e il modello grande prende come input l'output dei sotto livelli

- \_inbound\_nodes
- \_input\_spec
- input_spec
- input
- _outbound_nodes

Usando \_outbound\_nodes sembra possibile accedere alle operazioni anche quando queste non appartengono al sotto modello. Posso ripercorrere il grafo al contrario (dall'input all'output).
In realtà il problema rimane: l'outbound nodes del livello che rappresenta il sotto modello ha queste informazioni, ma non l'ultimo livello del sotto modello, quindi sto da capo a 12...

Ricapitolando:
1. Posso avere un modello che contiene dei sotto modelli
2. Se il modello contiene un sotto modello, il layer che rappresenta questo sub model è trattato come un layer singolo, non in modo composto
	1. Di conseguenza le keras_history e gli outbound_nodes sono tracciati considerando questo sotto modello come un livello unico

TODO. Rivedere il parsing nel complesso: ho bisogno anche di sapere da quale tensore di output del livello precedente deriva (questo succede se il livello precedente è a sua volta un keras.Model)


Ricostruzione di un modello unico:
```python
nextOpsDict
inputList <- init with input info
outputList <- init with output info

opQueue = []
opQueue <- init with entry info

while opQueue :
	currOp <- queue.getNext()
	if currOp is Model:
		## Manage Model
		subModelConns, subModelInputs, subModelOutputs <- recursive call
		
		Unify nextOpsDict and subModelsConns
		
		for elem in subModelOutpus:
			set next as next of currOp in main model
		
	else :
		## Manage Op
		get outbound nodes
		for elem in outbound_nodes :
			if elem is a subModel:
				change with identity layer with same name as sub model input layer
			else :
				add normal input
		update nextOpsDict
		
		
```
Problema con:
- Possibili operazioni non commutative
- Operazioni con costanti non accettate nell'input

Altro modo per gestire questi aspetti:
```python
def unwrapModel(model) :
	allOps <- findAllOps(model)
	executedOps : set[opNames] = {}
	tensorNamesMap
	for op in allOps :
		for prevOp in op.prevOps :
			if prevOp not in executedOps:
				call exec op on prevOp
		
		inputs = op._inbound_nodes[0].
		arguments._flat_arguments
		## This gives back a list of arguments
		newInputs = []
		for inp in inputs :
			if inp is kerasTensor :
				## All Predecessors have been computed yet
				newInputs.append(producedOutputs[inp])
			else :
				## Constant or something else --> Same
				newInputs.append(inp)
		opOutput = op.call(*newInputs)
		for elem, i in op.output:
			tensorNamesMap[elem.name] = opOutput[i]
		producedOutputs[opName] = opOutput	
		
```
Per fare questo mi serve:
- mappa kerasTensorName -> Operazione che lo genera
- mappa original keras tensor name -> new keras tensor
	- Quando lo rieseguo vengono prodotti dei nuovi nomi

CONTROLLARE CHE:
- \_inbound_nodes sia sempre ad un elemento
- Possibile problema con gli IdentityLayer inseriti per sostituire gli input layer del sotto modello
	- Loro non sanno il nome originale del tensore di output del livello di input che stanno sostituendo
	- Inoltre non sono mai stati chiamati, quindi non hanno degli inbound nodes!!
		- Posso fare un check per vedere se hanno gli attributi che sto cercando

Per accedere ai predecessori del sub model NON SO PERCHé, ma input non funziona, però
`_inbound_nodes[0].arguments._flat_arguments` sembra contenerli...


## Split del Modello (Post Unnest)
Alla luce di quanto capito del funzionamento di keras con l'unnest si è modificato lo split del modello.

Questo pezzo per la ricerca dell'output del modello non funziona: un livello potrebbe essere di output anche se ha dei successori!!
```python
if len(nextOpsDict[opName]) == 0:
## Output Layer
	for _, out in enumerate(layerOutput):
		subModelOutput[f"output_{idx}"] = out
		idx += 1

### CORRECT TO

if opName in model.output_names: ### List of output layers for the model
	find the tensor that is output of the model
	set it as model output
```


Per risolvere il problema dei nomi degli InputLayer dei sotto modelli creati, è stato inserito questo blocco post creazione del modello. Questo ci permette di rimuovere i suffissi CLONE che vengono aggiunti in automatico da Keras.
```python
for op in subModel.operations:
	opName: str = op.name
	if opName.endswith("CLONE"):
		opName = opName.removesuffix("CLONE")
	elif opName.endswith("clone"):
		opName = opName.removesuffix("clone")
	op.name = opName
```


| Pre Blocco      | ![[Schermata del 2024-12-14 19-08-58.png]] |
| --------------- | ------------------------------------------ |
| **Post Blocco** | ![[Schermata del 2024-12-14 19-09-50.png]] |
